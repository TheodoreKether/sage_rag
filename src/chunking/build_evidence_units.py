"""CLI entry point for Evidence Unit construction."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from .evidence_units import main
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.chunking.evidence_units import main

if __name__ == "__main__":
    sys.exit(main())
