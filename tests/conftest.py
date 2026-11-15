"""Gemeinsame Einrichtung: der Wurzelordner der Demo steht im Importpfad; `zufall(seed)` liefert den SplitMix64-Generator der Demo (nie Python-`random` oder `hash()`)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lyc_rng import SplitMix64  # noqa: E402


@pytest.fixture
def zufall():
    """Fabrik: `zufall(seed)` gibt einen frischen SplitMix64 mit diesem Zufalls-Seed."""
    return SplitMix64
