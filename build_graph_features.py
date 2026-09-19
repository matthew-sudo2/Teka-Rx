"""CLI script to build and save graph-derived features for TekaRx."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tekarx.transform.graph_features import build_graph_features


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract and aggregate graph relational features for TekaRx."
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("data/processed/tekarx_graph_arrays"),
        help="Path to graph arrays directory containing manifest.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/tekarx_graph_features.parquet"),
        help="Destination parquet file path.",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=3,
        help="Number of label propagation iterations.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Label propagation retention factor.",
    )
    args = parser.parse_args()

    print(f"Building graph features from {args.graph_dir}...")
    df = build_graph_features(
        graph_dir=args.graph_dir,
        output_path=args.output,
        n_iterations=args.n_iterations,
        alpha=args.alpha,
    )
    print(f"Graph features successfully built: shape={df.shape}")
    print(f"Saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
