"""Train the leakage-safe inductive TekaRx patient-drug GNN."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from tekarx.modeling import train_inductive_gnn


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(os.getenv("TEKARX_DATA_DIR", "data"))
    )
    parser.add_argument("--graph-path", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", help="torch device such as cuda, cuda:0, or cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-chunk-size", type=int, default=250_000)
    parser.add_argument(
        "--feature-track",
        choices=("prospective", "prospective-no-dosage", "completed_report"),
        default="prospective",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="explicitly consume the final test labels after model selection",
    )
    args = parser.parse_args()
    record = train_inductive_gnn(
        data_dir=args.data_dir,
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
    )
    print(json.dumps(asdict(record), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
