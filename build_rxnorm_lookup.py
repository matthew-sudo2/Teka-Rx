"""Convenience wrapper for `tekarx build-rxnorm-lookup`."""

from __future__ import annotations

import sys

from tekarx.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build-rxnorm-lookup", *sys.argv[1:]]))
