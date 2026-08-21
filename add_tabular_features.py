"""Convenience wrapper for `tekarx add-tabular-features`."""

from __future__ import annotations

import sys

from tekarx.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["add-tabular-features", *sys.argv[1:]]))
