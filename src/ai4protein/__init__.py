"""
AI4Protein: Utilities for protein AI course

This package provides reusable utilities for the AI for Protein course (KAIST Spring 2026).

Submodules:
- data: PDB loading, sequence utilities, graph construction
- models: Mini implementations of AlphaFold, RFDiffusion, ProteinMPNN
- utils: Visualization, metrics, geometry helpers
"""

__version__ = "0.1.0"

from . import data
from . import models
from . import utils
