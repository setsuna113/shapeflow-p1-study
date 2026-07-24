"""Make ``tests/fixtures`` importable from every test directory.

The scripted doubles (an author, a search backend, an engine) are shared between the unit and
integration suites. Duplicating them would let the two suites drift apart, and a double that
behaves differently in the test that guards a boundary from the one that exercises it is worse
than no double at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
