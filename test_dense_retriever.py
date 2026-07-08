"""Project-root CLI wrapper for dense retriever testing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.test_dense_retriever import main

if __name__ == "__main__":
    sys.exit(main())
