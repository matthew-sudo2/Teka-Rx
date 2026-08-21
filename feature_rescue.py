"""Convenience wrapper for `tekarx feature-rescue`."""

from __future__ import annotations

import sys

from tekarx.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["feature-rescue", *sys.argv[1:]]))
