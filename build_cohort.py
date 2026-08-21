"""Convenience wrapper for `tekarx build-cohort`."""

from __future__ import annotations

import sys

from tekarx.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build-cohort", *sys.argv[1:]]))
