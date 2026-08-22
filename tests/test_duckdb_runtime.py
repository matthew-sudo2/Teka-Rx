from pathlib import Path

import duckdb
import pytest

from tekarx.transform.duckdb_runtime import configure_duckdb


def test_configure_duckdb_bounds_memory_and_enables_local_spill(tmp_path: Path) -> None:
    connection = duckdb.connect()
    try:
        spill = configure_duckdb(
            connection,
            data_dir=tmp_path,
            stage="unit-test",
            memory_limit="512MB",
            threads=1,
        )
        preserve_order, configured_threads, temporary = connection.execute(
            "SELECT current_setting('preserve_insertion_order'), "
            "current_setting('threads'), current_setting('temp_directory')"
        ).fetchone()
    finally:
        connection.close()

    assert spill == tmp_path / "interim" / ".duckdb_temp" / "unit-test"
    assert spill.is_dir()
    assert preserve_order is False
    assert int(configured_threads) == 1
    assert Path(temporary) == spill


@pytest.mark.parametrize("stage", ["../escape", "space name", ""])
def test_configure_duckdb_rejects_unsafe_stage_names(tmp_path: Path, stage: str) -> None:
    connection = duckdb.connect()
    try:
        with pytest.raises(ValueError, match="invalid DuckDB stage name"):
            configure_duckdb(
                connection,
                data_dir=tmp_path,
                stage=stage,
                memory_limit="512MB",
                threads=1,
            )
    finally:
        connection.close()


def test_configure_duckdb_rejects_nonpositive_threads(tmp_path: Path) -> None:
    connection = duckdb.connect()
    try:
        with pytest.raises(ValueError, match="threads must be at least 1"):
            configure_duckdb(
                connection,
                data_dir=tmp_path,
                stage="unit-test",
                memory_limit="512MB",
                threads=0,
            )
    finally:
        connection.close()
