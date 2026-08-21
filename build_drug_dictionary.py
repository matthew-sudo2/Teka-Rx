"""Convenience wrapper for `tekarx build-drug-dictionary`."""

from __future__ import annotations

import sys

from tekarx.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build-drug-dictionary", *sys.argv[1:]]))
