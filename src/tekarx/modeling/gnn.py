"""Memory-bounded inductive patient-drug graph neural network training."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import random
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from tekarx.transform.graph import (
    AUXILIARY_TARGETS,
    NORMALIZED_DOSAGE_PATIENT_FEATURES,
    binary_auc,
)

if TYPE_CHECKING:
    import torch


class GNNTrainError(RuntimeError):
    """Raised when a graph cannot be trained under the inductive protocol."""


PROSPECTIVE_EXCLUDED_FEATURES = {
    "reporter_md",
    "reporter_cn",
    "reporter_hp",
    "reporter_ph",
    "reporter_lw",
    "reporter_unknown",
}
FEATURE_TRACKS = ("prospective", "prospective-no-dosage", "completed_report")
GNN_CHECKPOINT_FORMAT = "tekarx.inductive_gnn_training_checkpoint"
GNN_CHECKPOINT_VERSION = 1
DEFAULT_CHECKPOINT_EVERY = 5


@dataclass(frozen=True)
class GNNTrainRecord:
    """Summary of an inductive GNN training run."""

    model_path: str
    manifest_path: str
    graph_path: str
    feature_track: str
    device: str
    patient_nodes: int
    drug_nodes: int
    exposure_edges: int
    patient_feature_count: int
    drug_feature_count: int
    epochs_trained: int
    best_epoch: int
    validation_auc: float
    test_auc: float | None
    graph_storage: str
    neighbor_storage: str


@dataclass
class _TrainingGraph:
    """Arrays required by the trainer, backed by tensors or read-only memmaps."""

    graph_owner: object
    storage_backend: str
    patient_x: object
    drug_x: object
    labels: object
    split_id: object
    edge_patient_index: object
    edge_drug_index: object
    feature_names: tuple[str, ...]
    patient_count: int
    drug_count: int
    edge_count: int
    manifest: dict[str, Any] | None = None


def train_inductive_gnn(
    *,
    data_dir: Path,
    graph_path: Path | None = None,
    epochs: int = 100,
    batch_size: int = 8192,
    hidden_channels: int = 64,
    dropout: float = 0.20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 15,
    device: str | None = None,
    seed: int = 42,
    edge_chunk_size: int = 250_000,
    evaluate_test: bool = False,
    feature_track: str = "prospective",
    checkpoint_path: Path | None = None,
    resume_from: Path | None = None,
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
) -> GNNTrainRecord:
    """Train a one-hop mean-aggregation GNN without held-out message leakage.

    Drug-neighbor features are aggregated independently for each patient. Only
    training labels drive optimization, validation AUC drives early stopping,
    and test labels remain untouched unless ``evaluate_test`` is explicitly set.
    The aggregation is chunked on CPU, so it does not require optional PyG
    sampling backends and is suitable for a modest consumer GPU.
    """
    torch, nn, _ = _torch_dependencies()
    _validate_hyperparameters(
        epochs=epochs,
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        dropout=dropout,
        learning_rate=learning_rate,
        patience=patience,
        edge_chunk_size=edge_chunk_size,
        checkpoint_every=checkpoint_every,
    )
    if feature_track not in FEATURE_TRACKS:
        raise ValueError(f"feature_track must be one of {FEATURE_TRACKS}")
    graph_path = graph_path or data_dir / "processed" / "tekarx_graph.pt"
    if not graph_path.is_file():
        raise GNNTrainError(f"missing graph artifact: {graph_path}")

    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = _artifact_prefix(feature_track)
    model_path = output_dir / f"{artifact_prefix}.pt"
    manifest_path = output_dir / f"{artifact_prefix}_manifest.json"
    resume_source = Path(resume_from).resolve() if resume_from is not None else None
    checkpoint_target = Path(
        checkpoint_path
        or resume_from
        or output_dir / f"{artifact_prefix}_training_checkpoint.pt"
    ).resolve()
    protected_paths = {graph_path.resolve(), model_path.resolve(), manifest_path.resolve()}
    if checkpoint_target in protected_paths:
        raise GNNTrainError(
            "training checkpoint path must differ from the graph, final model, and manifest"
        )

    _seed_everything(torch, seed)
    training_graph = _load_training_graph(
        graph_path, validation_chunk_size=edge_chunk_size, torch=torch
    )
    selected_indices, feature_names = _select_feature_track(
        training_graph.feature_names, feature_track=feature_track
    )

    requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        training_device = torch.device(requested_device)
    except (RuntimeError, ValueError) as exc:
        raise GNNTrainError(f"invalid torch device: {requested_device}") from exc
    if training_device.type == "cuda" and not torch.cuda.is_available():
        raise GNNTrainError("CUDA was requested but is not available")
    graph_fingerprint = _graph_fingerprint(graph_path, training_graph)
    checkpoint_configuration = _checkpoint_configuration(
        graph_fingerprint=graph_fingerprint,
        feature_track=feature_track,
        feature_names=feature_names,
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        dropout=dropout,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        seed=seed,
        edge_chunk_size=edge_chunk_size,
        device_type=training_device.type,
    )
    resume_payload = None
    if resume_source is not None:
        resume_payload = _load_training_checkpoint(
            resume_source,
            expected_configuration=checkpoint_configuration,
            maximum_epoch=epochs,
            torch=torch,
        )
        print(
            f"Resuming after epoch {resume_payload['completed_epoch']} from "
            f"{resume_source}",
            flush=True,
        )
    temporary_neighbor_path: Path | None = None
    drug_neighbors: object | None = None
    try:
        print("Aggregating one-hop drug neighborhoods in bounded CPU chunks...", flush=True)
        if training_graph.storage_backend == "numpy_memmap_v1":
            descriptor_dir = graph_path.resolve().parent
            handle, temporary_name = tempfile.mkstemp(
                prefix=".gnn-neighbors-", suffix=".npy", dir=descriptor_dir
            )
            os.close(handle)
            temporary_neighbor_path = Path(temporary_name)
            drug_neighbors = _aggregate_drug_neighbors_memmap(
                patient_count=training_graph.patient_count,
                drug_features=training_graph.drug_x,
                edge_patient_index=training_graph.edge_patient_index,
                edge_drug_index=training_graph.edge_drug_index,
                output_path=temporary_neighbor_path,
                edge_chunk_size=edge_chunk_size,
                torch=torch,
            )
            neighbor_storage = "temporary_numpy_memmap"
        else:
            drug_neighbors = aggregate_drug_neighbors_from_arrays(
                patient_count=training_graph.patient_count,
                drug_features=training_graph.drug_x,
                edge_patient_index=training_graph.edge_patient_index,
                edge_drug_index=training_graph.edge_drug_index,
                edge_chunk_size=edge_chunk_size,
            )
            neighbor_storage = "cpu_tensor"

        if resume_payload is None:
            print(
                "Computing training-only normalization statistics in bounded chunks...",
                flush=True,
            )
            patient_mean, patient_std = _chunked_train_standardization(
                training_graph.patient_x,
                training_graph.split_id,
                feature_indices=selected_indices,
                chunk_size=edge_chunk_size,
                torch=torch,
                progress_label="Patient normalization",
            )
            neighbor_mean, neighbor_std = _chunked_train_standardization(
                drug_neighbors,
                training_graph.split_id,
                feature_indices=None,
                chunk_size=edge_chunk_size,
                torch=torch,
                progress_label="Neighbor normalization",
            )
        else:
            normalization = resume_payload["normalization"]
            patient_mean = normalization["patient_mean"]
            patient_std = normalization["patient_std"]
            neighbor_mean = normalization["neighbor_mean"]
            neighbor_std = normalization["neighbor_std"]

        patient_channels = len(selected_indices)
        drug_channels = _matrix_shape(drug_neighbors)[1]
        _validate_normalization_dimensions(
            patient_mean=patient_mean,
            patient_std=patient_std,
            neighbor_mean=neighbor_mean,
            neighbor_std=neighbor_std,
            patient_channels=patient_channels,
            drug_channels=drug_channels,
            torch=torch,
        )
        model = _make_model(
            nn,
            patient_channels=patient_channels,
            drug_channels=drug_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        ).to(training_device)
        optimizer = torch.optim.AdamW(
            model.parameters(), learning_rate, weight_decay=weight_decay
        )
        positives, negatives = _training_label_counts(
            training_graph.labels,
            training_graph.split_id,
            chunk_size=edge_chunk_size,
            torch=torch,
        )
        if positives == 0 or negatives == 0:
            raise GNNTrainError("training split must contain both target classes")
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(negatives / positives, device=training_device)
        )

        if resume_payload is None:
            best_auc = float("-inf")
            best_epoch = 0
            best_state: dict[str, object] | None = None
            stale_epochs = 0
            history: list[dict[str, float | int]] = []
            completed_epoch = 0
        else:
            model.load_state_dict(resume_payload["model_state_dict"])
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            _optimizer_to_device(optimizer, training_device, torch=torch)
            best_auc = float(resume_payload["best_auc"])
            best_epoch = int(resume_payload["best_epoch"])
            best_state = resume_payload["best_state_dict"]
            stale_epochs = int(resume_payload["stale_epochs"])
            history = copy.deepcopy(resume_payload["history"])
            completed_epoch = int(resume_payload["completed_epoch"])
            _restore_rng_state(resume_payload["rng_state"], training_device, torch=torch)

        for epoch in range(completed_epoch + 1, epochs + 1):
            if stale_epochs >= patience:
                print(
                    f"Checkpoint already satisfied early stopping at epoch "
                    f"{completed_epoch}.",
                    flush=True,
                )
                break
            model.train()
            epoch_loss = 0.0
            observed = 0
            generator = torch.Generator().manual_seed(seed + epoch)
            for batch_ids in _iter_split_batches(
                training_graph.split_id,
                split_value=0,
                row_count=training_graph.patient_count,
                scan_chunk_size=edge_chunk_size,
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
                torch=torch,
            ):
                optimizer.zero_grad(set_to_none=True)
                batch_patient = _normalized_matrix_batch(
                    training_graph.patient_x,
                    batch_ids,
                    feature_indices=selected_indices,
                    mean=patient_mean,
                    std=patient_std,
                    torch=torch,
                )
                batch_drugs = _normalized_matrix_batch(
                    drug_neighbors,
                    batch_ids,
                    feature_indices=None,
                    mean=neighbor_mean,
                    std=neighbor_std,
                    torch=torch,
                )
                batch_labels = _gather_vector(
                    training_graph.labels, batch_ids, dtype=torch.float32, torch=torch
                )
                if training_device.type == "cuda":
                    batch_patient = batch_patient.pin_memory()
                    batch_drugs = batch_drugs.pin_memory()
                    batch_labels = batch_labels.pin_memory()
                batch_patient = batch_patient.to(
                    training_device, non_blocking=training_device.type == "cuda"
                )
                batch_drugs = batch_drugs.to(
                    training_device, non_blocking=training_device.type == "cuda"
                )
                batch_labels = batch_labels.to(
                    training_device, non_blocking=training_device.type == "cuda"
                )
                logits = model(batch_patient, batch_drugs)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * batch_ids.numel()
                observed += batch_ids.numel()

            validation_scores, validation_labels = _predict_split(
                model,
                patient_x=training_graph.patient_x,
                drug_neighbors=drug_neighbors,
                labels=training_graph.labels,
                split_id=training_graph.split_id,
                split_value=1,
                patient_feature_indices=selected_indices,
                patient_mean=patient_mean,
                patient_std=patient_std,
                neighbor_mean=neighbor_mean,
                neighbor_std=neighbor_std,
                row_count=training_graph.patient_count,
                scan_chunk_size=edge_chunk_size,
                batch_size=batch_size,
                device=training_device,
                torch=torch,
            )
            validation_auc = binary_auc(validation_labels, validation_scores)
            mean_loss = epoch_loss / max(observed, 1)
            history.append(
                {"epoch": epoch, "train_loss": mean_loss, "validation_auc": validation_auc}
            )
            print(
                f"Epoch {epoch:03d}: train_loss={mean_loss:.6f} "
                f"validation_auc={validation_auc:.6f}",
                flush=True,
            )
            if validation_auc > best_auc + 1e-6:
                best_auc = validation_auc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            should_stop = stale_epochs >= patience
            if epoch == 1 or epoch % checkpoint_every == 0 or epoch == epochs or should_stop:
                _write_training_checkpoint(
                    checkpoint_target,
                    configuration=checkpoint_configuration,
                    completed_epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    best_state=best_state,
                    best_auc=best_auc,
                    best_epoch=best_epoch,
                    stale_epochs=stale_epochs,
                    history=history,
                    patient_mean=patient_mean,
                    patient_std=patient_std,
                    neighbor_mean=neighbor_mean,
                    neighbor_std=neighbor_std,
                    training_device=training_device,
                    torch=torch,
                )
                print(
                    f"Saved resumable checkpoint after epoch {epoch}: "
                    f"{checkpoint_target}",
                    flush=True,
                )
            if should_stop:
                break

        if best_state is None:
            raise GNNTrainError("training produced no checkpoint")
        model.load_state_dict(best_state)
        test_auc: float | None = None
        if evaluate_test:
            test_scores, test_labels = _predict_split(
                model,
                patient_x=training_graph.patient_x,
                drug_neighbors=drug_neighbors,
                labels=training_graph.labels,
                split_id=training_graph.split_id,
                split_value=2,
                patient_feature_indices=selected_indices,
                patient_mean=patient_mean,
                patient_std=patient_std,
                neighbor_mean=neighbor_mean,
                neighbor_std=neighbor_std,
                row_count=training_graph.patient_count,
                scan_chunk_size=edge_chunk_size,
                batch_size=batch_size,
                device=training_device,
                torch=torch,
            )
            test_auc = binary_auc(test_labels, test_scores)

        temporary_model = model_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "model_state_dict": best_state,
                "architecture": {
                    "name": "InductivePatientDrugMeanGNN",
                    "patient_channels": patient_channels,
                    "drug_channels": drug_channels,
                    "hidden_channels": hidden_channels,
                    "dropout": dropout,
                },
                "normalization": {
                    "patient_mean": patient_mean,
                    "patient_std": patient_std,
                    "drug_neighbor_mean": neighbor_mean,
                    "drug_neighbor_std": neighbor_std,
                },
                "patient_feature_names": feature_names,
                "best_epoch": best_epoch,
                "validation_auc": best_auc,
                "test_auc": test_auc,
                "seed": seed,
            },
            temporary_model,
        )
        os.replace(temporary_model, model_path)

        record = GNNTrainRecord(
            model_path=str(model_path),
            manifest_path=str(manifest_path),
            graph_path=str(graph_path),
            feature_track=feature_track,
            device=str(training_device),
            patient_nodes=training_graph.patient_count,
            drug_nodes=training_graph.drug_count,
            exposure_edges=training_graph.edge_count,
            patient_feature_count=patient_channels,
            drug_feature_count=drug_channels,
            epochs_trained=len(history),
            best_epoch=best_epoch,
            validation_auc=best_auc,
            test_auc=test_auc,
            graph_storage=training_graph.storage_backend,
            neighbor_storage=neighbor_storage,
        )
        _write_manifest(
            manifest_path,
            record=record,
            history=history,
            feature_names=feature_names,
            evaluate_test=evaluate_test,
            feature_track=feature_track,
            edge_chunk_size=edge_chunk_size,
            checkpoint_path=checkpoint_target,
            resumed_from=resume_source,
            checkpoint_every=checkpoint_every,
        )
        print(f"Best validation AUC: {best_auc:.6f} (epoch {best_epoch})", flush=True)
        if test_auc is None:
            print(
                "Test labels were not evaluated; the final test split remains untouched.",
                flush=True,
            )
        else:
            print(f"Explicit final test AUC: {test_auc:.6f}", flush=True)
        return record
    finally:
        if temporary_neighbor_path is not None:
            _close_memmap(drug_neighbors)
            drug_neighbors = None
            gc.collect()
            temporary_neighbor_path.unlink(missing_ok=True)


def _artifact_prefix(feature_track: str) -> str:
    prefixes = {
        "prospective": "tekarx_inductive_gnn",
        "prospective-no-dosage": "tekarx_no_dosage_inductive_gnn",
        "completed_report": "tekarx_completed_report_inductive_gnn",
    }
    return prefixes[feature_track]


def _graph_fingerprint(graph_path: Path, graph: _TrainingGraph) -> str:
    """Return a portable identity used to reject a checkpoint for another graph."""
    digest = hashlib.sha256()
    if graph.manifest is not None:
        identity = {
            "storage_backend": graph.storage_backend,
            "patient_count": graph.patient_count,
            "drug_count": graph.drug_count,
            "edge_count": graph.edge_count,
            "feature_names": graph.feature_names,
            "manifest": graph.manifest,
        }
        digest.update(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        return digest.hexdigest()

    with graph_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_configuration(
    *,
    graph_fingerprint: str,
    feature_track: str,
    feature_names: tuple[str, ...],
    batch_size: int,
    hidden_channels: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
    edge_chunk_size: int,
    device_type: str,
) -> dict[str, object]:
    return {
        "graph_fingerprint": graph_fingerprint,
        "feature_track": feature_track,
        "feature_names": list(feature_names),
        "batch_size": batch_size,
        "hidden_channels": hidden_channels,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "seed": seed,
        "edge_chunk_size": edge_chunk_size,
        "device_type": device_type,
    }


def _load_training_checkpoint(
    path: Path,
    *,
    expected_configuration: dict[str, object],
    maximum_epoch: int,
    torch: object,
) -> dict[str, object]:
    if not path.is_file():
        raise GNNTrainError(f"missing training checkpoint: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise GNNTrainError(f"cannot load training checkpoint: {path}") from exc
    if not isinstance(payload, dict):
        raise GNNTrainError("training checkpoint must contain a mapping")
    if payload.get("format") != GNN_CHECKPOINT_FORMAT:
        raise GNNTrainError(
            f"unsupported training checkpoint format: {payload.get('format')!r}"
        )
    if payload.get("format_version") != GNN_CHECKPOINT_VERSION:
        raise GNNTrainError(
            "unsupported training checkpoint version: "
            f"{payload.get('format_version')!r}"
        )
    if payload.get("test_evaluated") is not False:
        raise GNNTrainError("training checkpoint does not preserve the locked-test protocol")
    actual_configuration = payload.get("configuration")
    if actual_configuration != expected_configuration:
        if isinstance(actual_configuration, dict):
            changed = sorted(
                key
                for key in set(actual_configuration) | set(expected_configuration)
                if actual_configuration.get(key) != expected_configuration.get(key)
            )
        else:
            changed = ["configuration"]
        raise GNNTrainError(
            "training checkpoint is incompatible with this run; changed fields: "
            + ", ".join(changed)
        )

    required = {
        "completed_epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "best_state_dict",
        "best_auc",
        "best_epoch",
        "stale_epochs",
        "history",
        "normalization",
        "rng_state",
    }
    missing = required - set(payload)
    if missing:
        raise GNNTrainError(f"training checkpoint is missing fields: {sorted(missing)}")
    completed_epoch = payload["completed_epoch"]
    if not isinstance(completed_epoch, int) or completed_epoch < 1:
        raise GNNTrainError("training checkpoint completed_epoch must be positive")
    if completed_epoch > maximum_epoch:
        raise GNNTrainError(
            f"checkpoint is at epoch {completed_epoch}, beyond requested epoch {maximum_epoch}"
        )
    history = payload["history"]
    if not isinstance(history, list) or len(history) != completed_epoch:
        raise GNNTrainError("training checkpoint history does not match completed_epoch")
    if any(
        not isinstance(row, dict) or row.get("epoch") != index
        for index, row in enumerate(history, start=1)
    ):
        raise GNNTrainError("training checkpoint history has invalid epoch ordering")
    best_epoch = payload["best_epoch"]
    if not isinstance(best_epoch, int) or not 1 <= best_epoch <= completed_epoch:
        raise GNNTrainError("training checkpoint best_epoch is invalid")
    stale_epochs = payload["stale_epochs"]
    if not isinstance(stale_epochs, int) or stale_epochs < 0:
        raise GNNTrainError("training checkpoint stale_epochs is invalid")
    normalization = payload["normalization"]
    if not isinstance(normalization, dict) or set(normalization) != {
        "patient_mean",
        "patient_std",
        "neighbor_mean",
        "neighbor_std",
    }:
        raise GNNTrainError("training checkpoint normalization state is invalid")
    for name, value in normalization.items():
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise GNNTrainError(f"training checkpoint {name} must be a vector tensor")
        if not bool(torch.isfinite(value).all()):
            raise GNNTrainError(f"training checkpoint {name} contains NaN or infinity")
    if not isinstance(payload["rng_state"], dict):
        raise GNNTrainError("training checkpoint RNG state is invalid")
    return payload


def _write_training_checkpoint(
    path: Path,
    *,
    configuration: dict[str, object],
    completed_epoch: int,
    model: object,
    optimizer: object,
    best_state: dict[str, object] | None,
    best_auc: float,
    best_epoch: int,
    stale_epochs: int,
    history: list[dict[str, float | int]],
    patient_mean: object,
    patient_std: object,
    neighbor_mean: object,
    neighbor_std: object,
    training_device: object,
    torch: object,
) -> None:
    if best_state is None:
        raise GNNTrainError("cannot checkpoint training before a validation result exists")
    payload = {
        "format": GNN_CHECKPOINT_FORMAT,
        "format_version": GNN_CHECKPOINT_VERSION,
        "test_evaluated": False,
        "configuration": copy.deepcopy(configuration),
        "completed_epoch": completed_epoch,
        "model_state_dict": _cpu_clone(model.state_dict(), torch=torch),
        "optimizer_state_dict": _cpu_clone(optimizer.state_dict(), torch=torch),
        "best_state_dict": _cpu_clone(best_state, torch=torch),
        "best_auc": float(best_auc),
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": copy.deepcopy(history),
        "normalization": {
            "patient_mean": _cpu_clone(patient_mean, torch=torch),
            "patient_std": _cpu_clone(patient_std, torch=torch),
            "neighbor_mean": _cpu_clone(neighbor_mean, torch=torch),
            "neighbor_std": _cpu_clone(neighbor_std, torch=torch),
        },
        "rng_state": _capture_rng_state(training_device, torch=torch),
    }
    _atomic_torch_save(payload, path, torch=torch)


def _atomic_torch_save(payload: object, path: Path, *, torch: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cpu_clone(value: object, *, torch: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item, torch=torch) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item, torch=torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item, torch=torch) for item in value)
    return copy.deepcopy(value)


def _capture_rng_state(training_device: object, *, torch: object) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": (
            torch.cuda.get_rng_state(training_device).cpu()
            if training_device.type == "cuda"
            else None
        ),
    }


def _restore_rng_state(
    state: dict[str, object], training_device: object, *, torch: object
) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise GNNTrainError("training checkpoint RNG state has unexpected fields")
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if training_device.type == "cuda":
            cuda_state = state["torch_cuda"]
            if not isinstance(cuda_state, torch.Tensor):
                raise TypeError("missing CUDA RNG tensor")
            torch.cuda.set_rng_state(cuda_state, training_device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise GNNTrainError("training checkpoint RNG state cannot be restored") from exc


def _optimizer_to_device(optimizer: object, device: object, *, torch: object) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


def _validate_normalization_dimensions(
    *,
    patient_mean: object,
    patient_std: object,
    neighbor_mean: object,
    neighbor_std: object,
    patient_channels: int,
    drug_channels: int,
    torch: object,
) -> None:
    expected = {
        "patient_mean": (patient_mean, patient_channels),
        "patient_std": (patient_std, patient_channels),
        "neighbor_mean": (neighbor_mean, drug_channels),
        "neighbor_std": (neighbor_std, drug_channels),
    }
    for name, (value, channels) in expected.items():
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (channels,):
            raise GNNTrainError(f"{name} does not match the selected graph features")
    if bool(torch.any(patient_std <= 0)) or bool(torch.any(neighbor_std <= 0)):
        raise GNNTrainError("normalization standard deviations must be positive")


def _load_training_graph(
    graph_path: Path, *, validation_chunk_size: int, torch: object
) -> _TrainingGraph:
    """Open a legacy HeteroData graph or the v1 disk-backed graph descriptor."""
    artifact = torch.load(graph_path, map_location="cpu", weights_only=False)
    if isinstance(artifact, dict) and artifact.get("format") == "tekarx.memmap_graph":
        try:
            from tekarx.transform.graph_storage import load_graph_arrays
        except ImportError as exc:  # pragma: no cover - guarded by the graph extra
            raise GNNTrainError("disk-backed graph loader is unavailable") from exc
        bundle = load_graph_arrays(graph_path, mmap_mode="r")
        arrays = bundle.arrays
        manifest = bundle.manifest
        required = {
            "patient_x",
            "patient_y",
            "patient_split_id",
            "drug_x",
            "edge_patient_index",
            "edge_drug_index",
        }
        missing = required - set(arrays)
        if missing:
            raise GNNTrainError(f"disk-backed graph is missing arrays: {sorted(missing)}")
        feature_names = tuple(
            manifest.get("patient_feature_names")
            or manifest.get("patient_features")
            or ()
        )
        patient_x = arrays["patient_x"]
        drug_x = arrays["drug_x"]
        labels = arrays["patient_y"]
        split_id = arrays["patient_split_id"]
        edge_patient = arrays["edge_patient_index"]
        edge_drug = arrays["edge_drug_index"]
        _validate_memmap_training_graph(
            patient_x=patient_x,
            drug_x=drug_x,
            labels=labels,
            split_id=split_id,
            edge_patient_index=edge_patient,
            edge_drug_index=edge_drug,
            feature_names=feature_names,
            chunk_size=validation_chunk_size,
        )
        return _TrainingGraph(
            graph_owner=bundle,
            storage_backend="numpy_memmap_v1",
            patient_x=patient_x,
            drug_x=drug_x,
            labels=labels,
            split_id=split_id,
            edge_patient_index=edge_patient,
            edge_drug_index=edge_drug,
            feature_names=feature_names,
            patient_count=int(patient_x.shape[0]),
            drug_count=int(drug_x.shape[0]),
            edge_count=int(edge_patient.shape[0]),
            manifest=manifest,
        )

    graph = artifact
    feature_names = tuple(graph["patient"].get("feature_names", ()))
    _validate_inductive_graph(
        graph, feature_names=feature_names, chunk_size=validation_chunk_size
    )
    patient_x = graph["patient"].x.detach().cpu()
    drug_x = graph["drug"].x.detach().cpu()
    if patient_x.dtype != torch.float32 or drug_x.dtype != torch.float32:
        raise GNNTrainError("legacy graph node features must use float32")
    labels = graph["patient"].y.detach().cpu()
    if "split_id" in graph["patient"]:
        split_id = graph["patient"].split_id.detach().cpu().to(torch.int8)
    else:
        split_id = torch.full((patient_x.shape[0],), -1, dtype=torch.int8)
        split_id[graph["patient"].train_mask.detach().cpu().bool()] = 0
        split_id[graph["patient"].val_mask.detach().cpu().bool()] = 1
        split_id[graph["patient"].test_mask.detach().cpu().bool()] = 2
    reverse_edges = graph[("drug", "taken_by", "patient")].edge_index.detach().cpu()
    return _TrainingGraph(
        graph_owner=graph,
        storage_backend="legacy_heterodata_cpu",
        patient_x=patient_x,
        drug_x=drug_x,
        labels=labels,
        split_id=split_id,
        edge_patient_index=reverse_edges[1],
        edge_drug_index=reverse_edges[0],
        feature_names=feature_names,
        patient_count=int(patient_x.shape[0]),
        drug_count=int(drug_x.shape[0]),
        edge_count=int(reverse_edges.shape[1]),
    )


def _validate_memmap_training_graph(
    *,
    patient_x: object,
    drug_x: object,
    labels: object,
    split_id: object,
    edge_patient_index: object,
    edge_drug_index: object,
    feature_names: tuple[str, ...],
    chunk_size: int,
) -> None:
    """Validate disk-backed arrays without materializing cohort-sized masks."""
    patient_shape = _matrix_shape(patient_x)
    drug_shape = _matrix_shape(drug_x)
    if len(patient_shape) != 2 or len(drug_shape) != 2:
        raise GNNTrainError("patient_x and drug_x must be two-dimensional")
    patient_count = patient_shape[0]
    edge_count = int(edge_patient_index.shape[0])
    if not feature_names or len(feature_names) != patient_shape[1]:
        raise GNNTrainError("patient feature metadata does not match patient_x")
    forbidden = set(feature_names) & set(AUXILIARY_TARGETS)
    if forbidden:
        raise GNNTrainError(f"auxiliary outcomes found in patient features: {sorted(forbidden)}")
    if labels.ndim != 1 or labels.shape[0] != patient_count:
        raise GNNTrainError("patient_y does not match patient_x")
    if split_id.ndim != 1 or split_id.shape[0] != patient_count:
        raise GNNTrainError("patient_split_id does not match patient_x")
    if edge_patient_index.ndim != 1 or edge_drug_index.ndim != 1:
        raise GNNTrainError("disk-backed edge arrays must be one-dimensional")
    if edge_drug_index.shape[0] != edge_count:
        raise GNNTrainError("disk-backed edge arrays have different lengths")

    split_counts = [0, 0, 0]
    for start in range(0, patient_count, chunk_size):
        stop = min(start + chunk_size, patient_count)
        patient_chunk = np.asarray(patient_x[start:stop])
        if not np.isfinite(patient_chunk).all():
            raise GNNTrainError("graph patient features contain NaN or infinity")
        local_split = np.asarray(split_id[start:stop])
        if not np.isin(local_split, (0, 1, 2)).all():
            raise GNNTrainError("patient split ids must be 0, 1, or 2")
        for value in range(3):
            split_counts[value] += int(np.count_nonzero(local_split == value))
    print(f"Validated {patient_count:,} patient rows.", flush=True)
    if any(count == 0 for count in split_counts):
        raise GNNTrainError("patient train, validation, and test splits must all be non-empty")
    for start in range(0, drug_shape[0], chunk_size):
        if not np.isfinite(np.asarray(drug_x[start : start + chunk_size])).all():
            raise GNNTrainError("graph drug features contain NaN or infinity")
    print(f"Validated {drug_shape[0]:,} drug rows.", flush=True)
    for start in range(0, edge_count, chunk_size):
        stop = min(start + chunk_size, edge_count)
        patients = np.asarray(edge_patient_index[start:stop])
        drugs = np.asarray(edge_drug_index[start:stop])
        if patients.size and (patients.min() < 0 or patients.max() >= patient_count):
            raise GNNTrainError("edge array contains an out-of-range patient node")
        if drugs.size and (drugs.min() < 0 or drugs.max() >= drug_shape[0]):
            raise GNNTrainError("edge array contains an out-of-range drug node")
    print(f"Validated {edge_count:,} exposure edges.", flush=True)


def aggregate_drug_neighbors(
    *,
    patient_count: int,
    drug_features: torch.Tensor,
    drug_to_patient_edge_index: torch.Tensor,
    edge_chunk_size: int = 250_000,
) -> torch.Tensor:
    """Mean-aggregate drug features for each patient without edge-sized tensors."""
    if drug_to_patient_edge_index.ndim != 2 or drug_to_patient_edge_index.shape[0] != 2:
        raise GNNTrainError("drug-to-patient edge_index must have shape [2, num_edges]")
    return aggregate_drug_neighbors_from_arrays(
        patient_count=patient_count,
        drug_features=drug_features,
        edge_patient_index=drug_to_patient_edge_index[1],
        edge_drug_index=drug_to_patient_edge_index[0],
        edge_chunk_size=edge_chunk_size,
    )


def aggregate_drug_neighbors_from_arrays(
    *,
    patient_count: int,
    drug_features: torch.Tensor,
    edge_patient_index: torch.Tensor,
    edge_drug_index: torch.Tensor,
    edge_chunk_size: int = 250_000,
) -> torch.Tensor:
    """In-memory aggregation for legacy graphs without copying edge_index."""
    import torch

    if patient_count < 1 or edge_chunk_size < 1:
        raise ValueError("patient_count and edge_chunk_size must be positive")
    features = drug_features.detach().cpu()
    if features.dtype != torch.float32:
        raise GNNTrainError("drug features must use float32")
    patients = edge_patient_index.detach().cpu()
    drugs = edge_drug_index.detach().cpu()
    if patients.ndim != 1 or drugs.ndim != 1 or patients.shape[0] != drugs.shape[0]:
        raise GNNTrainError("edge patient/drug arrays must be equally sized vectors")
    output = torch.zeros((patient_count, features.shape[1]), dtype=torch.float32)
    degree = torch.zeros(patient_count, dtype=torch.float32)
    for start in range(0, patients.shape[0], edge_chunk_size):
        stop = min(start + edge_chunk_size, patients.shape[0])
        drug_ids = drugs[start:stop].long()
        patient_ids = patients[start:stop].long()
        if drug_ids.numel() and (
            int(drug_ids.min()) < 0 or int(drug_ids.max()) >= features.shape[0]
        ):
            raise GNNTrainError("edge_index contains an out-of-range drug node")
        if patient_ids.numel() and (
            int(patient_ids.min()) < 0 or int(patient_ids.max()) >= patient_count
        ):
            raise GNNTrainError("edge_index contains an out-of-range patient node")
        output.index_add_(0, patient_ids, features[drug_ids])
        degree.index_add_(0, patient_ids, torch.ones(patient_ids.numel(), dtype=torch.float32))
    orphan_count = 0
    for start in range(0, patient_count, edge_chunk_size):
        stop = min(start + edge_chunk_size, patient_count)
        local_degree = degree[start:stop]
        orphan_count += int(torch.count_nonzero(local_degree == 0))
        if not orphan_count:
            output[start:stop].div_(local_degree.unsqueeze(1))
    if orphan_count:
        raise GNNTrainError(f"graph contains {orphan_count} patients without edges")
    return output


def _aggregate_drug_neighbors_memmap(
    *,
    patient_count: int,
    drug_features: object,
    edge_patient_index: object,
    edge_drug_index: object,
    output_path: Path,
    edge_chunk_size: int,
    torch: object,
) -> np.memmap:
    """Aggregate into a disk-backed NPY array, keeping only one edge chunk resident."""
    if patient_count < 1 or edge_chunk_size < 1:
        raise ValueError("patient_count and edge_chunk_size must be positive")
    drug_count, drug_channels = _matrix_shape(drug_features)
    if edge_patient_index.shape[0] != edge_drug_index.shape[0]:
        raise GNNTrainError("edge patient/drug arrays have different lengths")
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(patient_count, drug_channels),
    )
    output[:] = 0.0
    output.flush()
    print(
        f"Initialized {patient_count:,} x {drug_channels:,} temporary neighbor matrix.",
        flush=True,
    )
    output_tensor = torch.from_numpy(output)
    degree = torch.zeros(patient_count, dtype=torch.float32)
    edge_count = int(edge_patient_index.shape[0])
    reported_percent = 0
    for start in range(0, edge_count, edge_chunk_size):
        stop = min(start + edge_chunk_size, edge_count)
        patient_ids = torch.from_numpy(
            np.array(edge_patient_index[start:stop], dtype=np.int64, copy=True)
        )
        drug_ids = torch.from_numpy(
            np.array(edge_drug_index[start:stop], dtype=np.int64, copy=True)
        )
        if patient_ids.numel() and (
            int(patient_ids.min()) < 0 or int(patient_ids.max()) >= patient_count
        ):
            raise GNNTrainError("edge array contains an out-of-range patient node")
        if drug_ids.numel() and (
            int(drug_ids.min()) < 0 or int(drug_ids.max()) >= drug_count
        ):
            raise GNNTrainError("edge array contains an out-of-range drug node")
        local_drugs = torch.from_numpy(
            np.ascontiguousarray(np.asarray(drug_features)[drug_ids.numpy()], dtype=np.float32)
        )
        output_tensor.index_add_(0, patient_ids, local_drugs)
        degree.index_add_(
            0, patient_ids, torch.ones(patient_ids.numel(), dtype=torch.float32)
        )
        reported_percent = _report_fractional_progress(
            "Neighbor edge aggregation",
            completed=stop,
            total=edge_count,
            reported_percent=reported_percent,
        )
    orphan_count = 0
    reported_percent = 0
    for start in range(0, patient_count, edge_chunk_size):
        stop = min(start + edge_chunk_size, patient_count)
        local_degree = degree[start:stop]
        local_orphans = int(torch.count_nonzero(local_degree == 0))
        orphan_count += local_orphans
        if local_orphans == 0:
            output_tensor[start:stop].div_(local_degree.unsqueeze(1))
        reported_percent = _report_fractional_progress(
            "Neighbor mean normalization",
            completed=stop,
            total=patient_count,
            reported_percent=reported_percent,
        )
    if orphan_count:
        raise GNNTrainError(f"graph contains {orphan_count} patients without edges")
    output.flush()
    del output_tensor, degree
    return output


def _matrix_shape(values: object) -> tuple[int, ...]:
    return tuple(int(value) for value in values.shape)


def _report_fractional_progress(
    label: str, *, completed: int, total: int, reported_percent: int
) -> int:
    if total <= 0:
        return reported_percent
    percentage = min(100, int(completed * 100 / total))
    milestone = 100 if completed >= total else (percentage // 10) * 10
    if milestone > reported_percent:
        print(f"{label}: {completed:,}/{total:,} ({percentage}%)", flush=True)
        return milestone
    return reported_percent


def _gather_matrix(
    values: object,
    row_ids: object,
    *,
    feature_indices: list[int] | None,
    torch: object,
) -> torch.Tensor:
    """Copy only the requested rows/columns from a tensor or memory map."""
    if isinstance(values, torch.Tensor):
        rows = row_ids.detach().cpu().long()
        if feature_indices is None:
            return values.index_select(0, rows).to(torch.float32)
        columns = torch.as_tensor(feature_indices, dtype=torch.long)
        return values[rows[:, None], columns[None, :]].to(torch.float32)
    rows_numpy = row_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    if feature_indices is None:
        result = np.array(values[rows_numpy], dtype=np.float32, copy=True, order="C")
    else:
        columns_numpy = np.asarray(feature_indices, dtype=np.int64)
        result = np.array(
            values[np.ix_(rows_numpy, columns_numpy)],
            dtype=np.float32,
            copy=True,
            order="C",
        )
    return torch.from_numpy(np.ascontiguousarray(result))


def _gather_vector(
    values: object, row_ids: object, *, dtype: object, torch: object
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.index_select(0, row_ids.detach().cpu().long()).to(dtype)
    rows = row_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    return torch.from_numpy(np.array(values[rows], copy=True, order="C")).to(dtype)


def _slice_split_ids(
    split_id: object, start: int, stop: int, *, torch: object
) -> torch.Tensor:
    if isinstance(split_id, torch.Tensor):
        return split_id[start:stop].detach().cpu().to(torch.int8)
    return torch.from_numpy(
        np.array(split_id[start:stop], dtype=np.int8, copy=True, order="C")
    )


def _iter_split_batches(
    split_id: object,
    *,
    split_value: int,
    row_count: int,
    scan_chunk_size: int,
    batch_size: int,
    shuffle: bool,
    generator: object | None,
    torch: object,
) -> Iterator[torch.Tensor]:
    """Yield block-shuffled split indices without a cohort-sized permutation."""
    block_count = (row_count + scan_chunk_size - 1) // scan_chunk_size
    if shuffle:
        block_order = torch.randperm(block_count, generator=generator).tolist()
    else:
        block_order = range(block_count)
    for block in block_order:
        start = block * scan_chunk_size
        stop = min(start + scan_chunk_size, row_count)
        local_split = _slice_split_ids(split_id, start, stop, torch=torch)
        local_ids = torch.nonzero(local_split == split_value, as_tuple=False).flatten()
        if local_ids.numel() == 0:
            continue
        local_ids.add_(start)
        if shuffle:
            permutation = torch.randperm(local_ids.numel(), generator=generator)
            local_ids = local_ids[permutation]
        for offset in range(0, local_ids.numel(), batch_size):
            yield local_ids[offset : offset + batch_size]


def _chunked_train_standardization(
    values: object,
    split_id: object,
    *,
    feature_indices: list[int] | None,
    chunk_size: int,
    torch: object,
    progress_label: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute stable train-only moments with O(chunk_size * features) memory."""
    row_count, feature_count = _matrix_shape(values)
    selected_count = feature_count if feature_indices is None else len(feature_indices)
    mean = torch.zeros(selected_count, dtype=torch.float64)
    second_moment = torch.zeros(selected_count, dtype=torch.float64)
    count = 0
    for batch_number, batch_ids in enumerate(
        _iter_split_batches(
            split_id,
            split_value=0,
            row_count=row_count,
            scan_chunk_size=chunk_size,
            batch_size=chunk_size,
            shuffle=False,
            generator=None,
            torch=torch,
        ),
        start=1,
    ):
        batch = _gather_matrix(
            values, batch_ids, feature_indices=feature_indices, torch=torch
        )
        if not bool(torch.isfinite(batch).all()):
            raise GNNTrainError("training features contain NaN or infinity")
        batch_var, batch_mean = torch.var_mean(batch, dim=0, unbiased=False)
        batch_mean = batch_mean.to(torch.float64)
        batch_second = batch_var.to(torch.float64) * batch.shape[0]
        batch_count = int(batch.shape[0])
        if count == 0:
            mean = batch_mean
            second_moment = batch_second
            count = batch_count
        else:
            total = count + batch_count
            delta = batch_mean - mean
            second_moment += batch_second + delta.square() * count * batch_count / total
            mean += delta * batch_count / total
            count = total
        if progress_label is not None and (batch_number == 1 or batch_number % 10 == 0):
            print(f"{progress_label}: processed {count:,} training rows", flush=True)
    if count == 0:
        raise GNNTrainError("training split is empty")
    if progress_label is not None:
        print(f"{progress_label}: complete ({count:,} training rows)", flush=True)
    variance = (second_moment / count).clamp_min(0.0)
    std = variance.sqrt()
    std = torch.where(std > 1e-6, std, torch.ones_like(std))
    return mean.to(torch.float32), std.to(torch.float32)


def _normalized_matrix_batch(
    values: object,
    row_ids: object,
    *,
    feature_indices: list[int] | None,
    mean: object,
    std: object,
    torch: object,
) -> torch.Tensor:
    batch = _gather_matrix(
        values, row_ids, feature_indices=feature_indices, torch=torch
    )
    batch.sub_(mean).div_(std)
    return batch


def _training_label_counts(
    labels: object,
    split_id: object,
    *,
    chunk_size: int,
    torch: object,
) -> tuple[int, int]:
    positives = 0
    total = 0
    row_count = int(labels.shape[0])
    for batch_ids in _iter_split_batches(
        split_id,
        split_value=0,
        row_count=row_count,
        scan_chunk_size=chunk_size,
        batch_size=chunk_size,
        shuffle=False,
        generator=None,
        torch=torch,
    ):
        batch = _gather_vector(labels, batch_ids, dtype=torch.int64, torch=torch)
        if not bool(((batch == 0) | (batch == 1)).all()):
            raise GNNTrainError("training labels must be binary")
        positives += int(batch.sum())
        total += int(batch.numel())
    return positives, total - positives


def _close_memmap(values: object | None) -> None:
    if isinstance(values, np.memmap):
        values.flush()
        values._mmap.close()


def _validate_inductive_graph(
    graph: object, *, feature_names: tuple[str, ...], chunk_size: int = 250_000
) -> None:
    import torch

    required_edge = ("drug", "taken_by", "patient")
    training_edge = ("patient", "takes", "drug")
    if required_edge not in graph.edge_types or training_edge not in graph.edge_types:
        raise GNNTrainError("graph is missing the required patient-drug relations")
    patient = graph["patient"]
    for name in ("x", "y", "train_mask", "val_mask", "test_mask"):
        if name not in patient:
            raise GNNTrainError(f"patient nodes are missing {name}")
    patient_count = int(patient.x.shape[0])
    split_counts = [0, 0, 0]
    for start in range(0, patient_count, chunk_size):
        stop = min(start + chunk_size, patient_count)
        local_masks = (
            patient.train_mask[start:stop],
            patient.val_mask[start:stop],
            patient.test_mask[start:stop],
        )
        local_sum = sum(mask.to(torch.int8) for mask in local_masks)
        if bool(torch.any(local_sum != 1)):
            raise GNNTrainError("patient split masks must be mutually exclusive and exhaustive")
        for index, mask in enumerate(local_masks):
            split_counts[index] += int(torch.count_nonzero(mask))
        if not bool(torch.isfinite(patient.x[start:stop]).all()):
            raise GNNTrainError("graph patient features contain NaN or infinity")
    if any(count == 0 for count in split_counts):
        raise GNNTrainError("patient train, validation, and test splits must all be non-empty")
    forbidden = set(feature_names) & set(AUXILIARY_TARGETS)
    if forbidden:
        raise GNNTrainError(f"auxiliary outcomes found in patient features: {sorted(forbidden)}")
    if not feature_names:
        raise GNNTrainError("graph is missing auditable patient feature metadata; rebuild it")
    if len(feature_names) != patient.x.shape[1]:
        raise GNNTrainError("patient feature metadata does not match patient.x")
    for start in range(0, graph["drug"].x.shape[0], chunk_size):
        if not bool(torch.isfinite(graph["drug"].x[start : start + chunk_size]).all()):
            raise GNNTrainError("graph drug features contain NaN or infinity")

    reverse = graph[required_edge]
    for name in ("train_mask", "val_mask", "test_mask"):
        if name not in reverse:
            raise GNNTrainError(f"drug-to-patient edges are missing {name}")
    forward = graph[training_edge].edge_index
    for start in range(0, forward.shape[1], chunk_size):
        if not bool(patient.train_mask[forward[0, start : start + chunk_size]].all()):
            raise GNNTrainError("held-out patients send messages into shared drug nodes")
    edge_count = int(reverse.edge_index.shape[1])
    train_edge_count = 0
    forward_offset = 0
    for start in range(0, edge_count, chunk_size):
        stop = min(start + chunk_size, edge_count)
        local_masks = (
            reverse.train_mask[start:stop],
            reverse.val_mask[start:stop],
            reverse.test_mask[start:stop],
        )
        local_sum = sum(mask.to(torch.int8) for mask in local_masks)
        if bool(torch.any(local_sum != 1)):
            raise GNNTrainError("edge split masks must be mutually exclusive and exhaustive")
        target_patients = reverse.edge_index[1, start:stop]
        expected_masks = (
            patient.train_mask[target_patients],
            patient.val_mask[target_patients],
            patient.test_mask[target_patients],
        )
        labels = ("training", "validation", "test")
        for label, actual, expected in zip(labels, local_masks, expected_masks, strict=True):
            if not torch.equal(actual, expected):
                raise GNNTrainError(
                    f"{label} edge mask does not match {label} patient targets"
                )
        local_train = local_masks[0]
        local_count = int(torch.count_nonzero(local_train))
        train_edge_count += local_count
        expected_forward = torch.stack(
            (
                reverse.edge_index[1, start:stop][local_train],
                reverse.edge_index[0, start:stop][local_train],
            )
        )
        actual_forward = forward[:, forward_offset : forward_offset + local_count]
        if not torch.equal(actual_forward, expected_forward):
            raise GNNTrainError(
                "patient-to-drug edges do not match the ordered training exposure topology"
            )
        forward_offset += local_count
    if forward.shape[1] != train_edge_count:
        raise GNNTrainError("patient-to-drug topology is not restricted to training edges")


def _select_feature_track(
    feature_names: tuple[str, ...], *, feature_track: str
) -> tuple[list[int], tuple[str, ...]]:
    if feature_track == "completed_report":
        return list(range(len(feature_names))), feature_names
    if feature_track not in FEATURE_TRACKS:
        raise ValueError(f"feature_track must be one of {FEATURE_TRACKS}")
    dosage_present = any(
        name in NORMALIZED_DOSAGE_PATIENT_FEATURES for name in feature_names
    )
    if feature_track == "prospective-no-dosage" and not dosage_present:
        raise GNNTrainError(
            "prospective-no-dosage requires a graph containing normalized dosage features"
        )
    selected = [
        index
        for index, name in enumerate(feature_names)
        if name not in PROSPECTIVE_EXCLUDED_FEATURES
        and not (
            feature_track == "prospective-no-dosage"
            and name in NORMALIZED_DOSAGE_PATIENT_FEATURES
        )
    ]
    if not selected:
        raise GNNTrainError("prospective feature selection removed every patient feature")
    return selected, tuple(feature_names[index] for index in selected)


def _make_model(
    nn: object,
    *,
    patient_channels: int,
    drug_channels: int,
    hidden_channels: int,
    dropout: float,
) -> object:
    class InductivePatientDrugMeanGNN(nn.Module):
        """GraphSAGE-style self/neighbor encoder for patient classification."""

        def __init__(self) -> None:
            super().__init__()
            self.patient_encoder = nn.Sequential(
                nn.Linear(patient_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
            )
            self.drug_encoder = nn.Sequential(
                nn.Linear(drug_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, 1),
            )

        def forward(self, patient_features: object, drug_neighbors: object) -> object:
            patient_hidden = self.patient_encoder(patient_features)
            drug_hidden = self.drug_encoder(drug_neighbors)
            return self.classifier(
                torch.cat((patient_hidden, drug_hidden), dim=1)
            ).squeeze(1)

    import torch

    return InductivePatientDrugMeanGNN()


def _split_count(
    split_id: object,
    *,
    split_value: int,
    row_count: int,
    chunk_size: int,
    torch: object,
) -> int:
    count = 0
    for start in range(0, row_count, chunk_size):
        stop = min(start + chunk_size, row_count)
        local = _slice_split_ids(split_id, start, stop, torch=torch)
        count += int(torch.count_nonzero(local == split_value))
    return count


def _predict_split(
    model: object,
    *,
    patient_x: object,
    drug_neighbors: object,
    labels: object,
    split_id: object,
    split_value: int,
    patient_feature_indices: list[int],
    patient_mean: object,
    patient_std: object,
    neighbor_mean: object,
    neighbor_std: object,
    row_count: int,
    scan_chunk_size: int,
    batch_size: int,
    device: object,
    torch: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Score one split; labels for other splits are never indexed or copied."""
    model.eval()
    output_count = _split_count(
        split_id,
        split_value=split_value,
        row_count=row_count,
        chunk_size=scan_chunk_size,
        torch=torch,
    )
    scores = np.empty(output_count, dtype=np.float32)
    truth = np.empty(output_count, dtype=np.int8)
    output_offset = 0
    with torch.no_grad():
        for batch_ids in _iter_split_batches(
            split_id,
            split_value=split_value,
            row_count=row_count,
            scan_chunk_size=scan_chunk_size,
            batch_size=batch_size,
            shuffle=False,
            generator=None,
            torch=torch,
        ):
            patient_batch = _normalized_matrix_batch(
                patient_x,
                batch_ids,
                feature_indices=patient_feature_indices,
                mean=patient_mean,
                std=patient_std,
                torch=torch,
            )
            neighbor_batch = _normalized_matrix_batch(
                drug_neighbors,
                batch_ids,
                feature_indices=None,
                mean=neighbor_mean,
                std=neighbor_std,
                torch=torch,
            )
            label_batch = _gather_vector(labels, batch_ids, dtype=torch.int8, torch=torch)
            if not bool(((label_batch == 0) | (label_batch == 1)).all()):
                raise GNNTrainError("evaluation labels must be binary")
            if device.type == "cuda":
                patient_batch = patient_batch.pin_memory()
                neighbor_batch = neighbor_batch.pin_memory()
            logits = model(
                patient_batch.to(device, non_blocking=device.type == "cuda"),
                neighbor_batch.to(device, non_blocking=device.type == "cuda"),
            )
            local_scores = torch.sigmoid(logits).cpu().numpy()
            stop = output_offset + batch_ids.numel()
            scores[output_offset:stop] = local_scores
            truth[output_offset:stop] = label_batch.numpy()
            output_offset = stop
    if output_offset != output_count:
        raise GNNTrainError("split scan produced an inconsistent row count")
    return scores, truth


def _torch_dependencies() -> tuple[object, object, object]:
    try:
        import torch
        import torch_geometric  # noqa: F401
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise GNNTrainError(
            'missing graph dependencies; install with python -m pip install -e ".[graph]"'
        ) from exc
    return torch, nn, DataLoader


def _seed_everything(torch: object, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_hyperparameters(
    *,
    epochs: int,
    batch_size: int,
    hidden_channels: int,
    dropout: float,
    learning_rate: float,
    patience: int,
    edge_chunk_size: int,
    checkpoint_every: int,
) -> None:
    if min(
        epochs,
        batch_size,
        hidden_channels,
        patience,
        edge_chunk_size,
        checkpoint_every,
    ) < 1:
        raise ValueError(
            "epochs, batch size, hidden size, patience, chunk size, and checkpoint "
            "interval must be positive"
        )
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")


def _write_manifest(
    path: Path,
    *,
    record: GNNTrainRecord,
    history: list[dict[str, float | int]],
    feature_names: tuple[str, ...],
    evaluate_test: bool,
    feature_track: str,
    edge_chunk_size: int,
    checkpoint_path: Path,
    resumed_from: Path | None,
    checkpoint_every: int,
) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "dataset": "TekaRx inductive patient-drug GNN",
                "intended_use": "research decision-support signal; not a diagnosis",
                "architecture": "one-hop mean drug-neighbor aggregation plus patient self features",
                "patient_features": list(feature_names),
                "feature_track": feature_track,
                "auxiliary_targets_in_x": False,
                "leakage_controls": {
                    "unknown_vocabulary": "selected from training patients only",
                    "optimization_labels": "training only",
                    "early_stopping": "validation only",
                    "test_evaluated": evaluate_test,
                    "held_out_patient_to_drug_messages": False,
                },
                "memory_strategy": {
                    "graph_storage": record.graph_storage,
                    "neighbor_storage": record.neighbor_storage,
                    "patient_feature_selection": "columns gathered only for the active minibatch",
                    "normalization_statistics": (
                        "training-only parallel moments combined in bounded row chunks"
                    ),
                    "normalization_application": (
                        "in place on copied minibatches; no cohort-sized normalized matrix"
                    ),
                    "training_shuffle": "random block order plus within-block permutation",
                    "edge_and_scan_chunk_size": edge_chunk_size,
                    "asymptotic_extra_ram": (
                        (
                            "O(patient_count + chunk_size * "
                            "(patient_features + drug_features)); the patient_count term is "
                            "the neighbor degree vector, not a feature matrix"
                        )
                        if record.neighbor_storage == "temporary_numpy_memmap"
                        else (
                            "O(patient_count * drug_features + chunk_size * patient_features); "
                            "legacy compatibility keeps one aggregated neighbor matrix in RAM"
                        )
                    ),
                },
                "resumability": {
                    "checkpoint_format": GNN_CHECKPOINT_FORMAT,
                    "checkpoint_version": GNN_CHECKPOINT_VERSION,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_every_epochs": checkpoint_every,
                    "resumed_from": str(resumed_from) if resumed_from is not None else None,
                    "checkpoint_test_evaluated": False,
                },
                "record": asdict(record),
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
