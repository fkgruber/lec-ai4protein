"""
3D geometry utilities for protein structures.
"""

from typing import Tuple, Optional
import numpy as np


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix.

    Args:
        coords: Coordinates of shape (N, 3)

    Returns:
        Distance matrix of shape (N, N)
    """
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def compute_dihedral(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray
) -> float:
    """
    Compute dihedral angle between four points.

    Args:
        p1, p2, p3, p4: 3D coordinates

    Returns:
        Dihedral angle in radians (-pi to pi)
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3

    # Normal vectors to planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    # Normalize
    n1 = n1 / (np.linalg.norm(n1) + 1e-10)
    n2 = n2 / (np.linalg.norm(n2) + 1e-10)

    # Calculate angle
    m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-10))

    x = np.dot(n1, n2)
    y = np.dot(m1, n2)

    return np.arctan2(y, x)


def compute_backbone_dihedrals(
    coords: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute backbone dihedral angles (phi, psi, omega).

    Args:
        coords: Backbone coordinates of shape (N, 4, 3) where
                dim 1 is [N, CA, C, O]

    Returns:
        Tuple of:
        - phi: (N,) array (first residue is nan)
        - psi: (N,) array (last residue is nan)
        - omega: (N,) array (first residue is nan)
    """
    n_residues = len(coords)

    phi = np.full(n_residues, np.nan)
    psi = np.full(n_residues, np.nan)
    omega = np.full(n_residues, np.nan)

    for i in range(n_residues):
        # phi: C(i-1) - N(i) - CA(i) - C(i)
        if i > 0:
            phi[i] = compute_dihedral(
                coords[i-1, 2],  # C(i-1)
                coords[i, 0],    # N(i)
                coords[i, 1],    # CA(i)
                coords[i, 2]     # C(i)
            )

        # psi: N(i) - CA(i) - C(i) - N(i+1)
        if i < n_residues - 1:
            psi[i] = compute_dihedral(
                coords[i, 0],    # N(i)
                coords[i, 1],    # CA(i)
                coords[i, 2],    # C(i)
                coords[i+1, 0]   # N(i+1)
            )

        # omega: CA(i-1) - C(i-1) - N(i) - CA(i)
        if i > 0:
            omega[i] = compute_dihedral(
                coords[i-1, 1],  # CA(i-1)
                coords[i-1, 2],  # C(i-1)
                coords[i, 0],    # N(i)
                coords[i, 1]     # CA(i)
            )

    return phi, psi, omega


def kabsch_align(
    mobile: np.ndarray,
    target: np.ndarray
) -> np.ndarray:
    """
    Align mobile coordinates to target using Kabsch algorithm.

    Args:
        mobile: Coordinates to align (N, 3)
        target: Target coordinates (N, 3)

    Returns:
        Aligned mobile coordinates (N, 3)
    """
    # Center both structures
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)

    mobile_centered = mobile - mobile_center
    target_centered = target - target_center

    # Compute optimal rotation using SVD
    H = mobile_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply rotation and translation
    aligned = mobile_centered @ R + target_center

    return aligned


def rotation_matrix_from_axis_angle(
    axis: np.ndarray,
    angle: float
) -> np.ndarray:
    """
    Compute rotation matrix from axis-angle representation.

    Args:
        axis: Unit rotation axis (3,)
        angle: Rotation angle in radians

    Returns:
        Rotation matrix (3, 3)
    """
    axis = axis / (np.linalg.norm(axis) + 1e-10)

    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c

    x, y, z = axis

    return np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
    ])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.

    Args:
        q: Quaternion [w, x, y, z]

    Returns:
        Rotation matrix (3, 3)
    """
    w, x, y, z = q / (np.linalg.norm(q) + 1e-10)

    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*w*z,      2*x*z + 2*w*y],
        [2*x*y + 2*w*z,      1 - 2*x*x - 2*z*z,  2*y*z - 2*w*x],
        [2*x*z - 2*w*y,      2*y*z + 2*w*x,      1 - 2*x*x - 2*y*y]
    ])


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion.

    Args:
        R: Rotation matrix (3, 3)

    Returns:
        Quaternion [w, x, y, z]
    """
    trace = np.trace(R)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z])


def compute_rigid_transform(
    coords1: np.ndarray,
    coords2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rigid transformation (rotation, translation) between coordinate sets.

    Args:
        coords1: Source coordinates (N, 3)
        coords2: Target coordinates (N, 3)

    Returns:
        Tuple of:
        - R: Rotation matrix (3, 3)
        - t: Translation vector (3,)
    """
    # Center both
    c1 = coords1.mean(axis=0)
    c2 = coords2.mean(axis=0)

    coords1_centered = coords1 - c1
    coords2_centered = coords2 - c2

    # Compute rotation using SVD
    H = coords1_centered.T @ coords2_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = c2 - R @ c1

    return R, t
