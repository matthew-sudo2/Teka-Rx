"""One-call orchestration for the prospective TekaRx data pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tekarx.studies import FAERS_PRESETS
from tekarx.transform.cohort import build_cohort
from tekarx.transform.drug_dictionary import build_drug_dictionary
from tekarx.transform.feature_rescue import build_feature_rescue
from tekarx.transform.tabular_features import add_tabular_features

_FAERS_TABLES = ("demo", "drug", "indi", "reac", "outc", "delete")
_DRUGCENTRAL_TABLES = ("structures", "synonyms", "struct2atc")


class ProspectivePipelineError(RuntimeError):
    """Raised when source artifacts required by the pipeline are unavailable."""


@dataclass(frozen=True)
class ProspectivePipelineRecord:
    """JSON-ready summaries returned by each prospective pipeline stage."""

    split_preset: str
    cohort: dict[str, Any]
    drug_dictionary: dict[str, Any]
    tabular_features: dict[str, Any]
    feature_rescue: dict[str, Any]


def build_prospective_pipeline(
    *,
    data_dir: Path,
    split_preset: str = "gnn-small",
    memory_limit: str = "4GB",
    threads: int | None = None,
    fuzzy_trigger_rate: float = 0.50,
    fuzzy_score_cutoff: float = 97.0,
    fuzzy_margin: float = 3.0,
    rebuild_graph: bool = True,
) -> ProspectivePipelineRecord:
    """Build the prospective cohort/features and optionally the final graph."""
    root = Path(data_dir)
    if split_preset not in FAERS_PRESETS:
        choices = ", ".join(sorted(FAERS_PRESETS))
        raise ValueError(f"unknown split preset {split_preset!r}; choose: {choices}")
    if threads is not None and threads < 1:
        raise ValueError("threads must be at least 1")
    _validate_prerequisites(root, split_preset=split_preset)

    cohort_record = build_cohort(
        data_dir=root,
        split_preset=split_preset,
        memory_limit=memory_limit,
        threads=threads,
    )
    dictionary_record = build_drug_dictionary(
        data_dir=root,
        fuzzy_trigger_rate=fuzzy_trigger_rate,
        fuzzy_score_cutoff=fuzzy_score_cutoff,
        fuzzy_margin=fuzzy_margin,
        memory_limit=memory_limit,
    )
    feature_record = add_tabular_features(
        data_dir=root,
        memory_limit=memory_limit,
        threads=threads,
        rebuild_graph=False,
    )
    rescue_record = build_feature_rescue(
        data_dir=root,
        memory_limit=memory_limit,
        threads=threads,
        rebuild_graph=rebuild_graph,
    )
    return ProspectivePipelineRecord(
        split_preset=split_preset,
        cohort=asdict(cohort_record),
        drug_dictionary=asdict(dictionary_record),
        tabular_features=asdict(feature_record),
        feature_rescue=asdict(rescue_record),
    )


def _validate_prerequisites(data_dir: Path, *, split_preset: str) -> None:
    missing: list[str] = []
    faers_root = data_dir / "interim" / "faers"
    for table in _FAERS_TABLES:
        if not any((faers_root / table).glob("*.parquet")):
            missing.append(f"FAERS {table}: {faers_root / table / '*.parquet'}")

    drugcentral_root = data_dir / "interim" / "drugcentral"
    for table in _DRUGCENTRAL_TABLES:
        path = drugcentral_root / f"{table}.parquet"
        if not path.is_file():
            missing.append(f"DrugCentral {table}: {path}")

    dumps = sorted((data_dir / "raw" / "drugcentral").glob("*.sql.gz"))
    if not dumps:
        missing.append(
            f"DrugCentral raw SQL dump: {data_dir / 'raw' / 'drugcentral' / '*.sql.gz'}"
        )
    elif len(dumps) > 1:
        names = ", ".join(str(path) for path in dumps)
        missing.append(f"exactly one DrugCentral raw SQL dump is required; found: {names}")

    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise ProspectivePipelineError(
            "prospective pipeline prerequisites are incomplete:\n"
            f"{details}\n"
            "Prepare them with:\n"
            f"  tekarx build-faers --preset {split_preset}\n"
            "  tekarx extract-drugcentral\n"
            "  tekarx build-drugcentral"
        )
