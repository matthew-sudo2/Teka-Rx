import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest

import tekarx.transform.pipeline as pipeline


@dataclass(frozen=True)
class _FakeRecord:
    stage: str
    rows: int


def _write_prerequisites(data_dir: Path) -> None:
    faers = data_dir / "interim" / "faers"
    for table in ("demo", "drug", "indi", "reac", "outc", "delete"):
        path = faers / table / "2023Q1.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    drugcentral = data_dir / "interim" / "drugcentral"
    drugcentral.mkdir(parents=True, exist_ok=True)
    for table in ("structures", "synonyms", "struct2atc"):
        (drugcentral / f"{table}.parquet").touch()
    raw = data_dir / "raw" / "drugcentral"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "drugcentral.sql.gz").touch()


def test_build_prospective_pipeline_runs_stages_in_order_and_returns_json_ready_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    _write_prerequisites(data_dir)
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_cohort(**kwargs: object) -> _FakeRecord:
        calls.append(("cohort", kwargs))
        return _FakeRecord("cohort", 10)

    def fake_dictionary(**kwargs: object) -> _FakeRecord:
        calls.append(("dictionary", kwargs))
        return _FakeRecord("dictionary", 20)

    def fake_features(**kwargs: object) -> _FakeRecord:
        calls.append(("features", kwargs))
        return _FakeRecord("features", 10)

    def fake_rescue(**kwargs: object) -> _FakeRecord:
        calls.append(("rescue", kwargs))
        return _FakeRecord("rescue", 10)

    monkeypatch.setattr(pipeline, "build_cohort", fake_cohort)
    monkeypatch.setattr(pipeline, "build_drug_dictionary", fake_dictionary)
    monkeypatch.setattr(pipeline, "add_tabular_features", fake_features)
    monkeypatch.setattr(pipeline, "build_feature_rescue", fake_rescue)

    record = pipeline.build_prospective_pipeline(
        data_dir=data_dir,
        split_preset="gnn-full",
        memory_limit="512MB",
        threads=2,
        rebuild_graph=True,
    )

    assert [stage for stage, _ in calls] == ["cohort", "dictionary", "features", "rescue"]
    assert calls[0][1]["split_preset"] == "gnn-full"
    assert calls[2][1]["rebuild_graph"] is False
    assert calls[3][1]["rebuild_graph"] is True
    payload = asdict(record)
    assert payload["cohort"] == {"stage": "cohort", "rows": 10}
    assert payload["drug_dictionary"] == {"stage": "dictionary", "rows": 20}
    assert payload["tabular_features"] == {"stage": "features", "rows": 10}
    assert payload["feature_rescue"] == {"stage": "rescue", "rows": 10}
    json.dumps(payload)


def test_build_prospective_pipeline_lists_all_missing_prerequisites(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"

    with pytest.raises(pipeline.ProspectivePipelineError) as error:
        pipeline.build_prospective_pipeline(data_dir=data_dir)

    message = str(error.value)
    assert "FAERS demo" in message
    assert "FAERS drug" in message
    assert "DrugCentral structures" in message
    assert "DrugCentral raw SQL dump" in message
    assert "tekarx build-faers --preset gnn-small" in message
