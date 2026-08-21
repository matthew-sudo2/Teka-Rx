"""Compare validation AUC with and without normalized dosage features."""

from __future__ import annotations

import sys

from tekarx.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluate-dosage-ablation", *sys.argv[1:]]))
