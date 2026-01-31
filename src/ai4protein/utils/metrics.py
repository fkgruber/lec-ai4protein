"""
Evaluation metrics for protein structure and sequence tasks.
"""

from typing import Optional, Tuple
import numpy as np


def compute_rmsd(
    coords1: np.ndarray,
    coords2: np.ndarray,
    align: bool = True
) -> float:
    """
    Compute Root Mean Square Deviation between two coordinate sets.

    Args:
        coords1: First coordinate set (N, 3)
        coords2: Second coordinate set (N, 3)
        align: Whether to perform Kabsch alignment first

    Returns:
        RMSD value in Angstroms
    """
    assert coords1.shape == coords2.shape, "Coordinate shapes must match"

    if align:
        from .geometry import kabsch_align
        coords2 = kabsch_align(coords2, coords1)

    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=-1)))


def compute_gdt_ts(
    coords1: np.ndarray,
    coords2: np.ndarray,
    thresholds: Tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
) -> float:
    """
    Compute Global Distance Test - Total Score.

    GDT-TS is the average percentage of residues within various
    distance thresholds after optimal superposition.

    Args:
        coords1: Reference coordinates (N, 3)
        coords2: Model coordinates (N, 3)
        thresholds: Distance thresholds in Angstroms

    Returns:
        GDT-TS score (0-100)
    """
    from .geometry import kabsch_align

    # Align coords2 to coords1
    coords2_aligned = kabsch_align(coords2, coords1)

    # Compute per-residue distances
    distances = np.sqrt(np.sum((coords1 - coords2_aligned) ** 2, axis=-1))

    # Compute percentage within each threshold
    percentages = []
    for threshold in thresholds:
        pct = np.mean(distances < threshold) * 100
        percentages.append(pct)

    return np.mean(percentages)


def compute_tm_score(
    coords1: np.ndarray,
    coords2: np.ndarray,
    l_target: Optional[int] = None
) -> float:
    """
    Compute Template Modeling Score (TM-score).

    TM-score is length-normalized and more sensitive to global
    topology than RMSD.

    Args:
        coords1: Reference coordinates (N, 3)
        coords2: Model coordinates (N, 3)
        l_target: Target length for normalization (default: len(coords1))

    Returns:
        TM-score (0-1)
    """
    from .geometry import kabsch_align

    n = len(coords1)
    if l_target is None:
        l_target = n

    # d0 is a length-dependent scale
    d0 = 1.24 * (l_target - 15) ** (1.0/3.0) - 1.8
    d0 = max(d0, 0.5)

    # Align coords2 to coords1
    coords2_aligned = kabsch_align(coords2, coords1)

    # Compute per-residue distances
    distances = np.sqrt(np.sum((coords1 - coords2_aligned) ** 2, axis=-1))

    # Compute TM-score
    tm_score = np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / l_target

    return tm_score


def sequence_recovery(
    predicted: str,
    native: str,
    ignore_unknown: bool = True
) -> float:
    """
    Compute sequence recovery rate.

    Args:
        predicted: Predicted amino acid sequence
        native: Native amino acid sequence
        ignore_unknown: Ignore positions with 'X' in native

    Returns:
        Recovery rate (0-1)
    """
    assert len(predicted) == len(native), "Sequences must have same length"

    matches = 0
    total = 0

    for p, n in zip(predicted.upper(), native.upper()):
        if ignore_unknown and n == 'X':
            continue
        total += 1
        if p == n:
            matches += 1

    return matches / total if total > 0 else 0.0


def compute_contact_precision(
    predicted: np.ndarray,
    native: np.ndarray,
    min_separation: int = 6,
    top_k: Optional[int] = None
) -> float:
    """
    Compute precision of predicted contacts.

    Args:
        predicted: Predicted contact probabilities (N, N)
        native: Native contact map (binary, N, N)
        min_separation: Minimum sequence separation for contacts
        top_k: Number of top predictions to consider (default: N)

    Returns:
        Precision score (0-1)
    """
    n = len(predicted)
    if top_k is None:
        top_k = n

    # Create separation mask
    sep_mask = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) >= min_separation

    # Get upper triangle (avoid counting twice)
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mask = sep_mask & triu_mask

    # Get top k predictions
    pred_masked = predicted.copy()
    pred_masked[~mask] = -np.inf

    flat_idx = np.argsort(pred_masked.flatten())[::-1][:top_k]
    rows = flat_idx // n
    cols = flat_idx % n

    # Compute precision
    true_positives = native[rows, cols].sum()
    precision = true_positives / top_k

    return precision


def compute_perplexity(
    log_probs: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute perplexity from log probabilities.

    Args:
        log_probs: Log probabilities of shape (N,) or (B, N)
        mask: Optional mask for valid positions

    Returns:
        Perplexity value
    """
    if mask is not None:
        log_probs = log_probs[mask]

    avg_nll = -np.mean(log_probs)
    return np.exp(avg_nll)


def compute_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Predicted class labels
        targets: True class labels
        mask: Optional mask for valid positions

    Returns:
        Accuracy (0-1)
    """
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]

    return np.mean(predictions == targets)
