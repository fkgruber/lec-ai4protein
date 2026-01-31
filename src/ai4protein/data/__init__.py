"""
Data loading and processing utilities for proteins.

Modules:
- pdb: PDB file loading and structure extraction
- sequence: Sequence encoding and manipulation
- graph: Graph construction from protein structures
"""

from .pdb import load_pdb, get_backbone_coords, get_sequence_from_structure
from .sequence import (
    one_hot_encode,
    blosum_encode,
    physicochemical_encode,
    encode_sequence,
)
from .graph import (
    build_knn_graph,
    build_contact_graph,
    structure_to_pyg_data,
)

__all__ = [
    "load_pdb",
    "get_backbone_coords",
    "get_sequence_from_structure",
    "one_hot_encode",
    "blosum_encode",
    "physicochemical_encode",
    "encode_sequence",
    "build_knn_graph",
    "build_contact_graph",
    "structure_to_pyg_data",
]
