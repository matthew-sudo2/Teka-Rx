"""Convenience wrapper for `tekarx build-prospective`."""

from __future__ import annotations

import sys

from tekarx.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build-prospective", *sys.argv[1:]]))
