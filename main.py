"""Repository-local entry point; installation is optional for quick experiments."""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from m2m_agda.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
