"""
Sequence encoding utilities for protein sequences.
"""

from typing import Optional, Literal
import numpy as np

# Standard amino acids
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# BLOSUM62 matrix (symmetric, only upper triangle stored)
BLOSUM62 = np.array([
    [ 4, 0,-2,-1,-2, 0,-2,-1,-1,-1,-1,-2,-1,-1,-1, 1, 0, 0,-3,-2],  # A
    [ 0, 9,-3,-4,-2,-3,-3,-1,-3,-1,-1,-3,-3,-3,-3,-1,-1,-1,-2,-2],  # C
    [-2,-3, 6, 2,-3,-1,-1,-3,-1,-4,-3, 1,-1, 0,-2, 0,-1,-3,-4,-3],  # D
    [-1,-4, 2, 5,-3,-2, 0,-3, 1,-3,-2, 0,-1, 2, 0, 0,-1,-2,-3,-2],  # E
    [-2,-2,-3,-3, 6,-3,-1, 0,-3, 0, 0,-3,-4,-3,-3,-2,-2,-1, 1, 3],  # F
    [ 0,-3,-1,-2,-3, 6,-2,-4,-2,-4,-3, 0,-2,-2,-2, 0,-2,-3,-2,-3],  # G
    [-2,-3,-1, 0,-1,-2, 8,-3,-1,-3,-2, 1,-2, 0, 0,-1,-2,-3,-2, 2],  # H
    [-1,-1,-3,-3, 0,-4,-3, 4,-3, 2, 1,-3,-3,-3,-3,-2,-1, 3,-3,-1],  # I
    [-1,-3,-1, 1,-3,-2,-1,-3, 5,-2,-1, 0,-1, 1, 2, 0,-1,-2,-3,-2],  # K
    [-1,-1,-4,-3, 0,-4,-3, 2,-2, 4, 2,-3,-3,-2,-2,-2,-1, 1,-2,-1],  # L
    [-1,-1,-3,-2, 0,-3,-2, 1,-1, 2, 5,-2,-2, 0,-1,-1,-1, 1,-1,-1],  # M
    [-2,-3, 1, 0,-3, 0, 1,-3, 0,-3,-2, 6,-2, 0, 0, 1, 0,-3,-4,-2],  # N
    [-1,-3,-1,-1,-4,-2,-2,-3,-1,-3,-2,-2, 7,-1,-2,-1,-1,-2,-4,-3],  # P
    [-1,-3, 0, 2,-3,-2, 0,-3, 1,-2, 0, 0,-1, 5, 1, 0,-1,-2,-2,-1],  # Q
    [-1,-3,-2, 0,-3,-2, 0,-3, 2,-2,-1, 0,-2, 1, 5,-1,-1,-3,-3,-2],  # R
    [ 1,-1, 0, 0,-2, 0,-1,-2, 0,-2,-1, 1,-1, 0,-1, 4, 1,-2,-3,-2],  # S
    [ 0,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1, 0,-1,-1,-1, 1, 5, 0,-2,-2],  # T
    [ 0,-1,-3,-2,-1,-3,-3, 3,-2, 1, 1,-3,-2,-2,-3,-2, 0, 4,-3,-1],  # V
    [-3,-2,-4,-3, 1,-2,-2,-3,-3,-2,-1,-4,-4,-2,-3,-3,-2,-3,11, 2],  # W
    [-2,-2,-3,-2, 3,-3, 2,-1,-2,-1,-1,-2,-3,-1,-2,-2,-2,-1, 2, 7],  # Y
], dtype=np.float32)

# Physicochemical properties (normalized)
# Columns: hydrophobicity, volume, charge, polarity, aromaticity
PHYSICOCHEMICAL = {
    'A': [ 0.62, 0.11, 0.0, 0.0, 0.0],
    'C': [ 0.29, 0.24, 0.0, 0.0, 0.0],
    'D': [-0.90, 0.27, -1.0, 1.0, 0.0],
    'E': [-0.74, 0.40, -1.0, 1.0, 0.0],
    'F': [ 1.19, 0.55, 0.0, 0.0, 1.0],
    'G': [ 0.48, 0.00, 0.0, 0.0, 0.0],
    'H': [-0.40, 0.43, 0.5, 1.0, 1.0],
    'I': [ 1.38, 0.45, 0.0, 0.0, 0.0],
    'K': [-1.50, 0.53, 1.0, 1.0, 0.0],
    'L': [ 1.06, 0.45, 0.0, 0.0, 0.0],
    'M': [ 0.64, 0.47, 0.0, 0.0, 0.0],
    'N': [-0.78, 0.32, 0.0, 1.0, 0.0],
    'P': [ 0.12, 0.26, 0.0, 0.0, 0.0],
    'Q': [-0.85, 0.43, 0.0, 1.0, 0.0],
    'R': [-2.53, 0.60, 1.0, 1.0, 0.0],
    'S': [-0.18, 0.14, 0.0, 1.0, 0.0],
    'T': [-0.05, 0.26, 0.0, 1.0, 0.0],
    'V': [ 1.08, 0.33, 0.0, 0.0, 0.0],
    'W': [ 0.81, 0.74, 0.0, 0.0, 1.0],
    'Y': [ 0.26, 0.60, 0.0, 1.0, 1.0],
}


def one_hot_encode(sequence: str, include_unknown: bool = True) -> np.ndarray:
    """
    One-hot encode an amino acid sequence.

    Args:
        sequence: Amino acid sequence string
        include_unknown: If True, add extra dimension for unknown residues

    Returns:
        One-hot encoded array of shape (L, 20) or (L, 21)
    """
    n_classes = 21 if include_unknown else 20
    encoding = np.zeros((len(sequence), n_classes), dtype=np.float32)

    for i, aa in enumerate(sequence.upper()):
        if aa in AA_TO_IDX:
            encoding[i, AA_TO_IDX[aa]] = 1.0
        elif include_unknown:
            encoding[i, 20] = 1.0  # Unknown residue

    return encoding


def blosum_encode(sequence: str) -> np.ndarray:
    """
    Encode sequence using BLOSUM62 substitution matrix rows.

    Each amino acid is represented by its row in BLOSUM62,
    which captures evolutionary substitution patterns.

    Args:
        sequence: Amino acid sequence string

    Returns:
        BLOSUM-encoded array of shape (L, 20)
    """
    encoding = np.zeros((len(sequence), 20), dtype=np.float32)

    for i, aa in enumerate(sequence.upper()):
        if aa in AA_TO_IDX:
            encoding[i] = BLOSUM62[AA_TO_IDX[aa]]
        # Unknown residues get zero vector

    # Normalize to [-1, 1] range
    encoding = encoding / 9.0

    return encoding


def physicochemical_encode(sequence: str) -> np.ndarray:
    """
    Encode sequence using physicochemical properties.

    Properties: hydrophobicity, volume, charge, polarity, aromaticity

    Args:
        sequence: Amino acid sequence string

    Returns:
        Property-encoded array of shape (L, 5)
    """
    encoding = np.zeros((len(sequence), 5), dtype=np.float32)

    for i, aa in enumerate(sequence.upper()):
        if aa in PHYSICOCHEMICAL:
            encoding[i] = PHYSICOCHEMICAL[aa]

    return encoding


def encode_sequence(
    sequence: str,
    method: Literal['onehot', 'blosum', 'physicochemical', 'combined'] = 'onehot'
) -> np.ndarray:
    """
    Encode a protein sequence using specified method.

    Args:
        sequence: Amino acid sequence string
        method: Encoding method
            - 'onehot': 21-dimensional one-hot encoding
            - 'blosum': 20-dimensional BLOSUM62 encoding
            - 'physicochemical': 5-dimensional property encoding
            - 'combined': Concatenation of all (46 dimensions)

    Returns:
        Encoded sequence array
    """
    if method == 'onehot':
        return one_hot_encode(sequence)
    elif method == 'blosum':
        return blosum_encode(sequence)
    elif method == 'physicochemical':
        return physicochemical_encode(sequence)
    elif method == 'combined':
        return np.concatenate([
            one_hot_encode(sequence),
            blosum_encode(sequence),
            physicochemical_encode(sequence),
        ], axis=1)
    else:
        raise ValueError(f"Unknown encoding method: {method}")


def sequence_to_indices(sequence: str) -> np.ndarray:
    """
    Convert sequence to integer indices.

    Args:
        sequence: Amino acid sequence

    Returns:
        Integer array of shape (L,) with values 0-19 (20 for unknown)
    """
    indices = np.zeros(len(sequence), dtype=np.int64)
    for i, aa in enumerate(sequence.upper()):
        indices[i] = AA_TO_IDX.get(aa, 20)
    return indices


def indices_to_sequence(indices: np.ndarray) -> str:
    """
    Convert integer indices back to sequence.

    Args:
        indices: Integer array with values 0-19

    Returns:
        Amino acid sequence string
    """
    return ''.join(AMINO_ACIDS[i] if i < 20 else 'X' for i in indices)
