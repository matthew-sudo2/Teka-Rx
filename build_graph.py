"""Convenience wrapper for `tekarx build-graph`."""

from __future__ import annotations

import sys

from tekarx.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build-graph", *sys.argv[1:]]))
