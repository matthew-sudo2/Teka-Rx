"""Command-line entry point for immutable source extraction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import NoReturn

from tekarx.extract import extract_dailymed, extract_drugcentral, extract_faers
from tekarx.extract.dailymed import DAILYMED_DATASETS
from tekarx.extract.drugcentral import DEFAULT_DRUGCENTRAL_URL
from tekarx.extract.faers import resolve_faers_source
from tekarx.modeling import evaluate_dosage_ablation, train_inductive_gnn
from tekarx.paths import DataPaths
from tekarx.studies import FAERS_PRESETS, preset_quarters, write_split_plan
from tekarx.transform import (
    CORE_DRUGCENTRAL_TABLES,
    CORE_FAERS_TABLES,
    add_tabular_features,
    build_cohort,
    build_dailymed,
    build_drug_dictionary,
    build_drugcentral,
    build_faers,
    build_feature_rescue,
    build_graph,
    build_prospective_pipeline,
    build_rxnorm_lookup,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tekarx", description="TekaRx source extraction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    faers = subparsers.add_parser("extract-faers", help="download a quarterly FAERS ASCII ZIP")
    _shared_arguments(faers)
    selection = faers.add_mutually_exclusive_group()
    selection.add_argument(
        "--quarter",
        help="quarter such as 2024Q1; omit to download the latest available quarter",
    )
    selection.add_argument(
        "--preset",
        choices=sorted(FAERS_PRESETS),
        help="download all quarters for a reproducible experiment split",
    )
    faers.add_argument("--url", help="advanced override for the official FDA ASCII ZIP URL")

    drugcentral = subparsers.add_parser(
        "extract-drugcentral", help="download the DrugCentral PostgreSQL dump"
    )
    _shared_arguments(drugcentral)
    drugcentral.add_argument("--url", default=DEFAULT_DRUGCENTRAL_URL)

    dailymed = subparsers.add_parser(
        "extract-dailymed", help="download a compact DailyMed mapping archive"
    )
    _shared_arguments(dailymed)
    dailymed.add_argument(
        "--dataset",
        choices=sorted(DAILYMED_DATASETS),
        help="omit to download all compact mapping datasets",
    )
    dailymed.add_argument("--url", help="override the official URL")

    build = subparsers.add_parser(
        "build-faers", help="stream extracted FAERS ASCII tables to Snappy Parquet"
    )
    build.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("TEKARX_DATA_DIR", "data")),
    )
    build_selection = build.add_mutually_exclusive_group(required=True)
    build_selection.add_argument("--quarter", help="build one quarter such as 2024Q1")
    build_selection.add_argument("--preset", choices=sorted(FAERS_PRESETS))
    build.add_argument(
        "--tables",
        nargs="+",
        choices=CORE_FAERS_TABLES,
        default=list(CORE_FAERS_TABLES),
        help="tables to convert; defaults to all core and deletion tables",
    )

    build_dc = subparsers.add_parser(
        "build-drugcentral", help="stream DrugCentral mapping tables to Snappy Parquet"
    )
    build_dc.add_argument(
        "--data-dir", type=Path, default=Path(os.getenv("TEKARX_DATA_DIR", "data"))
    )
    build_dc.add_argument(
        "--tables",
        nargs="+",
        choices=CORE_DRUGCENTRAL_TABLES,
        default=list(CORE_DRUGCENTRAL_TABLES),
        help="mapping tables to convert; defaults to all mapping-relevant tables",
    )

    build_dm = subparsers.add_parser(
        "build-dailymed", help="stream DailyMed mapping archives to Snappy Parquet"
    )
    build_dm.add_argument(
        "--data-dir", type=Path, default=Path(os.getenv("TEKARX_DATA_DIR", "data"))
    )
    build_dm.add_argument(
        "--dataset",
        choices=sorted(DAILYMED_DATASETS),
        help="omit to build all downloaded compact mapping datasets",
    )

    cohort = subparsers.add_parser(
        "build-cohort", help="build the latest-version FAERS cohort and graph edges"
    )
    cohort.add_argument("--data-dir", type=Path, default=Path(os.getenv("TEKARX_DATA_DIR", "data")))
    cohort.add_argument(
        "--split-preset",
        choices=sorted(FAERS_PRESETS),
        default="gnn-small",
        help="exact quarter-based split definition; default: gnn-small",
    )
    cohort.add_argument(
        "--memory-limit",
        default="4GB",
        help="DuckDB memory limit before spilling to disk; default: 4GB",
    )
    cohort.add_argument("--threads", type=int, help="optional DuckDB worker-thread limit")

    dictionary = subparsers.add_parser(
        "build-drug-dictionary", help="map FAERS names and calculate drug-level ROR"
    )
    dictionary.add_argument(
        "--data-dir", type=Path, default=Path(os.getenv("TEKARX_DATA_DIR", "data"))
    )
    dictionary.add_argument("--fuzzy-trigger-rate", type=float, default=0.50)
    dictionary.add_argument("--fuzzy-score-cutoff", type=float, default=97.0)
    dictionary.add_argument("--fuzzy-margin", type=float, default=3.0)
    dictionary.add_argument("--memory-limit", default="4GB")
    dictionary.add_argument("--threads", type=int, help="optional DuckDB worker-thread limit")

    rxnorm = subparsers.add_parser(
        "build-rxnorm-lookup",
        help="build a DrugCentral RxNorm name lookup and optionally query RxNav",
    )
    rxnorm.add_argument(
        "--data-dir", type=Path, default=Path(os.getenv("TEKARX_DATA_DIR", "data"))
    )
    rxnorm.add_argument(
        "--use-api",
        action="store_true",
        help="query RxNav for currently unmapped FAERS names; disabled by default",
    )
    rxnorm.add_argument("--batch-size", type=int, default=250)
    rxnorm.add_argument("--requests-per-second", type=float, default=10.0)
    rxnorm.add_argument(
        "--max-names",
        type=int,
        help="optional cap for testing or incremental API runs",
    )

    graph = subparsers.add_parser(
        "build-graph", help="build PyG HeteroData and the XGBoost tabular baseline"
    )
    graph.add_argument(
        "--data-dir", type=Path, default=Path(os.getenv("TEKARX_DATA_DIR", "data"))
    )
    graph.add_argument(
        "--cohort-path",
        type=Path,
        help="explicit cohort Parquet; use the feature-rescue cohort for the final model",
    )
    graph.add_argument("--top-unknown", type=int, default=500)
    graph.add_argument("--memory-limit", default="4GB")
    graph.add_argument("--threads", type=int)
    graph.add_argument("--xgb-rounds", type=int, default=500)
    graph.add_argument("--xgb-early-stopping", type=int, default=30)
    graph.add_argument("--xgb-max-depth", type=int, default=5)
    graph.add_argument("--xgb-max-leaves", type=int, default=0)
    graph.add_argument(
        "--xgb-device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="XGBoost training device; auto uses CUDA only when both Torch and XGBoost support it",
    )
    graph.add_argument(
        "--graph-storage",
        choices=("memory-mapped", "legacy"),
        default="memory-mapped",
        help="store one-copy mmap arrays (default) or a memory-heavy legacy PyG pickle",
    )
    graph.add_argument("--materialization-batch-size", type=int, default=131_072)
    graph.add_argument("--xgb-batch-size", type=int, default=65_536)

    dosage_ablation = subparsers.add_parser(
        "evaluate-dosage-ablation",
        help="compare XGBoost validation AUC with and without normalized dosage",
    )
    dosage_ablation.add_argument(
        "--data-dir", type=Path, default=Path(os.getenv("TEKARX_DATA_DIR", "data"))
    )
    dosage_ablation.add_argument("--tabular-path", type=Path)
    dosage_ablation.add_argument("--xgb-rounds", type=int, default=1000)
    dosage_ablation.add_argument("--xgb-early-stopping", type=int, default=50)
    dosage_ablation.add_argument("--xgb-max-depth", type=int, default=0)
    dosage_ablation.add_argument("--xgb-max-leaves", type=int, default=63)
    dosage_ablation.add_argument("--threads", type=int)
    dosage_ablation.add_argument("--seed", type=int, default=42)
    dosage_ablation.add_argument("--bootstrap-samples", type=int, default=500)

    features = subparsers.add_parser(
        "add-tabular-features", help="enrich the cohort and rebuild graph/baseline artifacts"
    )
    features.add_argument(
        "--data-dir", type=Path, default=Path(os.getenv("TEKARX_DATA_DIR", "data"))
    )
    features.add_argument("--memory-limit", default="4GB")
    features.add_argument("--threads", type=int)
    features.add_argument(
        "--skip-graph",
        action="store_true",
        help="write enriched dosage artifacts without rebuilding graph/model artifacts",
    )

    rescue = subparsers.add_parser(
        "feature-rescue", help="fit train-frozen indication/pair features and retrain models"
    )
    rescue.add_argument(
        "--data-dir", type=Path, default=Path(os.getenv("TEKARX_DATA_DIR", "data"))
    )
    rescue.add_argument("--training-end-year", type=int, default=2023)
    rescue.add_argument("--top-pairs", type=int, default=50)
    rescue.add_argument("--minimum-pair-reports", type=int, default=25)
    rescue.add_argument("--memory-limit", default="4GB")
    rescue.add_argument("--threads", type=int)
    rescue.add_argument(
        "--skip-graph",
        action="store_true",
        help="write train-frozen rescue artifacts without rebuilding graph/model artifacts",
    )

    prospective = subparsers.add_parser(
        "build-prospective",
        help="run the corrected cohort, dictionary, feature, and graph pipeline",
    )
    prospective.add_argument(
        "--data-dir", type=Path, default=Path(os.getenv("TEKARX_DATA_DIR", "data"))
    )
    prospective.add_argument(
        "--split-preset",
        choices=sorted(FAERS_PRESETS),
        default="gnn-small",
        help="exact quarter-based split definition; default: gnn-small",
    )
    prospective.add_argument("--memory-limit", default="4GB")
    prospective.add_argument("--threads", type=int)
    prospective.add_argument("--fuzzy-trigger-rate", type=float, default=0.50)
    prospective.add_argument("--fuzzy-score-cutoff", type=float, default=97.0)
    prospective.add_argument("--fuzzy-margin", type=float, default=3.0)
    prospective.add_argument(
        "--skip-graph",
        action="store_true",
        help="build the enriched cohort without rebuilding graph/model artifacts",
    )

    gnn = subparsers.add_parser(
        "train-gnn", help="train the leakage-safe inductive patient-drug GNN"
    )
    gnn.add_argument("--data-dir", type=Path, default=Path(os.getenv("TEKARX_DATA_DIR", "data")))
    gnn.add_argument("--graph-path", type=Path)
    gnn.add_argument("--epochs", type=int, default=100)
    gnn.add_argument("--batch-size", type=int, default=8192)
    gnn.add_argument("--hidden-channels", type=int, default=64)
    gnn.add_argument("--dropout", type=float, default=0.20)
    gnn.add_argument("--learning-rate", type=float, default=1e-3)
    gnn.add_argument("--weight-decay", type=float, default=1e-4)
    gnn.add_argument("--patience", type=int, default=15)
    gnn.add_argument("--device", help="torch device such as cuda, cuda:0, or cpu")
    gnn.add_argument("--seed", type=int, default=42)
    gnn.add_argument("--edge-chunk-size", type=int, default=250_000)
    gnn.add_argument(
        "--checkpoint-path",
        type=Path,
        help=(
            "atomic training-state checkpoint destination; defaults to the processed "
            "directory"
        ),
    )
    gnn.add_argument(
        "--resume-from",
        type=Path,
        help="resume an interrupted run from a compatible training checkpoint",
    )
    gnn.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="save resumable training state every N epochs (default: 5)",
    )
    gnn.add_argument(
        "--feature-track",
        choices=("prospective", "prospective-no-dosage", "completed_report"),
        default="prospective",
    )
    gnn.add_argument(
        "--evaluate-test",
        action="store_true",
        help="explicitly consume final test labels after model selection",
    )
    return parser


def _shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("TEKARX_DATA_DIR", "data")),
    )
    parser.add_argument("--sha256", help="optional expected SHA-256 checksum")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = DataPaths(args.data_dir)
    paths.create()
    try:
        if args.command == "extract-faers":
            if args.preset:
                if args.url or args.sha256:
                    raise ValueError("--url and --sha256 cannot be used with --preset")
                split_quarters = preset_quarters(args.preset)
                print(
                    f"Selected FAERS preset {args.preset}: {', '.join(split_quarters)}",
                    file=sys.stderr,
                )
                records = []
                for quarter in split_quarters:
                    resolved_quarter, url = resolve_faers_source(quarter=quarter, url=None)
                    print(f"Extracting FAERS {resolved_quarter}: {url}", file=sys.stderr)
                    records.append(
                        extract_faers(
                            data_dir=paths.root,
                            quarter=resolved_quarter,
                            url=url,
                        )
                    )
                split_path = write_split_plan(data_dir=paths.root, name=args.preset)
                print(
                    json.dumps(
                        {
                            "artifacts": [asdict(item) for item in records],
                            "split_plan": str(split_path),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0
            quarter, url = resolve_faers_source(quarter=args.quarter, url=args.url)
            print(f"Selected FAERS {quarter}: {url}", file=sys.stderr)
            record = extract_faers(
                data_dir=paths.root,
                quarter=quarter,
                url=url,
                expected_sha256=args.sha256,
            )
        elif args.command == "extract-drugcentral":
            record = extract_drugcentral(
                data_dir=paths.root,
                url=args.url,
                expected_sha256=args.sha256,
            )
        elif args.command == "extract-dailymed":
            if args.dataset is None and (args.url or args.sha256):
                raise ValueError("--url and --sha256 require a single --dataset")
            datasets = [args.dataset] if args.dataset else sorted(DAILYMED_DATASETS)
            records = [
                extract_dailymed(
                    data_dir=paths.root,
                    dataset=dataset,
                    url=args.url,
                    expected_sha256=args.sha256,
                )
                for dataset in datasets
            ]
            print(json.dumps([asdict(item) for item in records], indent=2, sort_keys=True))
            return 0
        elif args.command == "build-faers":
            quarters = preset_quarters(args.preset) if args.preset else [args.quarter.upper()]
            records = build_faers(
                data_dir=paths.root,
                quarters=quarters,
                tables=tuple(args.tables),
            )
            print(json.dumps([asdict(item) for item in records], indent=2, sort_keys=True))
            return 0
        elif args.command == "build-drugcentral":
            records = build_drugcentral(data_dir=paths.root, tables=tuple(args.tables))
            print(json.dumps([asdict(item) for item in records], indent=2, sort_keys=True))
            return 0
        elif args.command == "build-dailymed":
            datasets = (args.dataset,) if args.dataset else tuple(sorted(DAILYMED_DATASETS))
            records = build_dailymed(data_dir=paths.root, datasets=datasets)
            print(json.dumps([asdict(item) for item in records], indent=2, sort_keys=True))
            return 0
        elif args.command == "build-cohort":
            cohort_record = build_cohort(
                data_dir=paths.root,
                split_preset=args.split_preset,
                memory_limit=args.memory_limit,
                threads=args.threads,
            )
            print(json.dumps(asdict(cohort_record), indent=2, sort_keys=True))
            return 0
        elif args.command == "build-drug-dictionary":
            dictionary_record = build_drug_dictionary(
                data_dir=paths.root,
                fuzzy_trigger_rate=args.fuzzy_trigger_rate,
                fuzzy_score_cutoff=args.fuzzy_score_cutoff,
                fuzzy_margin=args.fuzzy_margin,
                memory_limit=args.memory_limit,
                threads=args.threads,
            )
            print(json.dumps(asdict(dictionary_record), indent=2, sort_keys=True))
            return 0
        elif args.command == "build-rxnorm-lookup":
            rxnorm_record = build_rxnorm_lookup(
                data_dir=paths.root,
                use_api=args.use_api,
                batch_size=args.batch_size,
                requests_per_second=args.requests_per_second,
                max_names=args.max_names,
            )
            print(json.dumps(asdict(rxnorm_record), indent=2, sort_keys=True))
            return 0
        elif args.command == "build-graph":
            graph_record = build_graph(
                data_dir=paths.root,
                top_unknown=args.top_unknown,
                memory_limit=args.memory_limit,
                threads=args.threads,
                xgb_rounds=args.xgb_rounds,
                xgb_early_stopping=args.xgb_early_stopping,
                xgb_max_depth=args.xgb_max_depth,
                xgb_max_leaves=args.xgb_max_leaves,
                xgb_device=args.xgb_device,
                cohort_path=args.cohort_path,
                storage_mode=args.graph_storage,
                materialization_batch_size=args.materialization_batch_size,
                xgb_batch_size=args.xgb_batch_size,
            )
            print(json.dumps(asdict(graph_record), indent=2, sort_keys=True))
            return 0
        elif args.command == "evaluate-dosage-ablation":
            ablation_record = evaluate_dosage_ablation(
                data_dir=paths.root,
                tabular_path=args.tabular_path,
                xgb_rounds=args.xgb_rounds,
                xgb_early_stopping=args.xgb_early_stopping,
                xgb_max_depth=args.xgb_max_depth,
                xgb_max_leaves=args.xgb_max_leaves,
                threads=args.threads,
                seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
            )
            print(json.dumps(asdict(ablation_record), indent=2, sort_keys=True))
            return 0
        elif args.command == "add-tabular-features":
            feature_record = add_tabular_features(
                data_dir=paths.root,
                memory_limit=args.memory_limit,
                threads=args.threads,
                rebuild_graph=not args.skip_graph,
            )
            print(json.dumps(asdict(feature_record), indent=2, sort_keys=True))
            return 0
        elif args.command == "feature-rescue":
            rescue_record = build_feature_rescue(
                data_dir=paths.root,
                training_end_year=args.training_end_year,
                top_pairs=args.top_pairs,
                minimum_pair_reports=args.minimum_pair_reports,
                memory_limit=args.memory_limit,
                threads=args.threads,
                rebuild_graph=not args.skip_graph,
            )
            print(json.dumps(asdict(rescue_record), indent=2, sort_keys=True))
            return 0
        elif args.command == "build-prospective":
            pipeline_record = build_prospective_pipeline(
                data_dir=paths.root,
                split_preset=args.split_preset,
                memory_limit=args.memory_limit,
                threads=args.threads,
                fuzzy_trigger_rate=args.fuzzy_trigger_rate,
                fuzzy_score_cutoff=args.fuzzy_score_cutoff,
                fuzzy_margin=args.fuzzy_margin,
                rebuild_graph=not args.skip_graph,
            )
            print(json.dumps(asdict(pipeline_record), indent=2, sort_keys=True))
            return 0
        elif args.command == "train-gnn":
            gnn_record = train_inductive_gnn(
                data_dir=paths.root,
                graph_path=args.graph_path,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_channels=args.hidden_channels,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                patience=args.patience,
                device=args.device,
                seed=args.seed,
                edge_chunk_size=args.edge_chunk_size,
                evaluate_test=args.evaluate_test,
                feature_track=args.feature_track,
                checkpoint_path=args.checkpoint_path,
                resume_from=args.resume_from,
                checkpoint_every=args.checkpoint_every,
            )
            print(json.dumps(asdict(gnn_record), indent=2, sort_keys=True))
            return 0
        else:
            _unreachable(args.command)
    except Exception as exc:
        parser = build_parser()
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(asdict(record), indent=2, sort_keys=True))
    return 0


def _unreachable(command: str) -> NoReturn:
    raise AssertionError(f"unhandled command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
