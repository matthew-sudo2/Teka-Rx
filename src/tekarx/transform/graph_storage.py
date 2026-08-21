"""Versioned, memory-mapped storage for large TekaRx graph arrays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MEMMAP_GRAPH_FORMAT = "tekarx.memmap_graph"
MEMMAP_GRAPH_VERSION = 1


class GraphStorageError(RuntimeError):
    """Raised when a memory-mapped graph descriptor is invalid."""


@dataclass(frozen=True)
class GraphArrayBundle:
    """Validated descriptor, manifest, and lazily memory-mapped numeric arrays."""

    descriptor_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]


def load_graph_arrays(
    descriptor_path: Path, *, mmap_mode: str | None = "r"
) -> GraphArrayBundle:
    """Load and validate a TekaRx memory-mapped graph descriptor.

    ``numpy.load`` is called with ``allow_pickle=False``.  With the default
    ``mmap_mode='r'`` the numeric payloads stay on disk and pages are faulted in
    only as a consumer touches them.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - graph extra owns torch
        raise GraphStorageError(
            'loading a graph descriptor requires python -m pip install -e ".[graph]"'
        ) from exc

    source = Path(descriptor_path).resolve()
    if not source.is_file():
        raise GraphStorageError(f"missing graph descriptor: {source}")
    try:
        descriptor = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise GraphStorageError(f"cannot load graph descriptor: {source}") from exc
    if not isinstance(descriptor, dict):
        raise GraphStorageError("graph artifact is a legacy HeteroData object, not a descriptor")
    if descriptor.get("format") != MEMMAP_GRAPH_FORMAT:
        raise GraphStorageError(f"unsupported graph format: {descriptor.get('format')!r}")
    if descriptor.get("format_version") != MEMMAP_GRAPH_VERSION:
        raise GraphStorageError(
            f"unsupported graph format version: {descriptor.get('format_version')!r}"
        )
    manifest_value = descriptor.get("manifest_path")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise GraphStorageError("graph descriptor has no manifest_path")
    manifest_path = (source.parent / manifest_value).resolve()
    if not manifest_path.is_file():
        raise GraphStorageError(f"missing graph array manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphStorageError(f"cannot read graph array manifest: {manifest_path}") from exc
    if manifest.get("format") != MEMMAP_GRAPH_FORMAT:
        raise GraphStorageError("descriptor and array manifest formats do not match")
    if manifest.get("format_version") != MEMMAP_GRAPH_VERSION:
        raise GraphStorageError("descriptor and array manifest versions do not match")

    raw_arrays = manifest.get("arrays")
    if not isinstance(raw_arrays, dict) or not raw_arrays:
        raise GraphStorageError("graph array manifest has no arrays")
    arrays: dict[str, np.ndarray] = {}
    array_root = manifest_path.parent.resolve()
    for name, metadata in raw_arrays.items():
        if not isinstance(metadata, dict):
            raise GraphStorageError(f"invalid metadata for graph array {name!r}")
        relative = metadata.get("path")
        if not isinstance(relative, str) or not relative:
            raise GraphStorageError(f"graph array {name!r} has no path")
        array_path = (array_root / relative).resolve()
        if array_root not in array_path.parents:
            raise GraphStorageError(f"graph array {name!r} escapes its artifact directory")
        if not array_path.is_file():
            raise GraphStorageError(f"missing graph array {name!r}: {array_path}")
        try:
            array = np.load(array_path, mmap_mode=mmap_mode, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise GraphStorageError(f"cannot load graph array {name!r}: {array_path}") from exc
        expected_dtype = metadata.get("dtype")
        expected_shape = metadata.get("shape")
        if str(array.dtype) != expected_dtype:
            raise GraphStorageError(
                f"graph array {name!r} dtype mismatch: {array.dtype} != {expected_dtype}"
            )
        if list(array.shape) != expected_shape:
            raise GraphStorageError(
                f"graph array {name!r} shape mismatch: {array.shape} != {expected_shape}"
            )
        arrays[name] = array
    return GraphArrayBundle(
        descriptor_path=source,
        manifest_path=manifest_path,
        manifest=manifest,
        arrays=arrays,
    )
