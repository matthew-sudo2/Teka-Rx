from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tekarx.load import write_parquet


def test_parquet_writer_is_snappy_and_atomic(tmp_path: Path) -> None:
    destination = tmp_path / "interim" / "table.parquet"
    write_parquet(pa.table({"id": [1, 2], "name": ["a", "b"]}), destination)

    parquet = pq.ParquetFile(destination)
    assert parquet.metadata.row_group(0).column(0).compression == "SNAPPY"
    assert not destination.with_suffix(".parquet.tmp").exists()
