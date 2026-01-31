"""
Graph construction utilities for protein structures.
"""

from typing import Optional, Tuple, Literal
import numpy as np


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distance matrix from coordinates.

    Args:
        coords: Atom coordinates of shape (N, 3)

    Returns:
        Distance matrix of shape (N, N)
    """
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def build_knn_graph(
    coords: np.ndarray,
    k: int = 10,
    self_loops: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build k-nearest neighbor graph from coordinates.

    Args:
        coords: Atom coordinates of shape (N, 3)
        k: Number of neighbors per node
        self_loops: Whether to include self-loops

    Returns:
        Tuple of:
        - edge_index: (2, E) array of edge indices
        - edge_attr: (E,) array of distances
    """
    n = len(coords)
    dist_matrix = compute_distance_matrix(coords)

    # For each node, find k nearest neighbors
    edges_src = []
    edges_dst = []
    edge_distances = []

    for i in range(n):
        distances = dist_matrix[i].copy()
        if not self_loops:
            distances[i] = np.inf  # Exclude self

        # Get k nearest
        k_actual = min(k, n - 1) if not self_loops else min(k, n)
        neighbors = np.argpartition(distances, k_actual)[:k_actual]

        for j in neighbors:
            edges_src.append(i)
            edges_dst.append(j)
            edge_distances.append(dist_matrix[i, j])

    edge_index = np.array([edges_src, edges_dst], dtype=np.int64)
    edge_attr = np.array(edge_distances, dtype=np.float32)

    return edge_index, edge_attr


def build_contact_graph(
    coords: np.ndarray,
    threshold: float = 8.0,
    self_loops: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build contact graph based on distance threshold.

    Args:
        coords: Atom coordinates of shape (N, 3)
        threshold: Distance threshold for contacts (Angstroms)
        self_loops: Whether to include self-loops

    Returns:
        Tuple of:
        - edge_index: (2, E) array of edge indices
        - edge_attr: (E,) array of distances
    """
    dist_matrix = compute_distance_matrix(coords)

    # Find contacts
    if self_loops:
        contacts = dist_matrix <= threshold
    else:
        contacts = (dist_matrix <= threshold) & (dist_matrix > 0)

    src, dst = np.where(contacts)
    edge_index = np.array([src, dst], dtype=np.int64)
    edge_attr = dist_matrix[src, dst].astype(np.float32)

    return edge_index, edge_attr


def build_sequential_graph(
    n_residues: int,
    window: int = 1
) -> np.ndarray:
    """
    Build sequential connectivity graph.

    Args:
        n_residues: Number of residues
        window: Connect residues within this window

    Returns:
        edge_index: (2, E) array of edge indices
    """
    edges_src = []
    edges_dst = []

    for i in range(n_residues):
        for j in range(max(0, i - window), min(n_residues, i + window + 1)):
            if i != j:
                edges_src.append(i)
                edges_dst.append(j)

    return np.array([edges_src, edges_dst], dtype=np.int64)


def structure_to_pyg_data(
    coords: np.ndarray,
    sequence: str,
    graph_type: Literal['knn', 'contact'] = 'knn',
    k: int = 10,
    threshold: float = 8.0,
    node_features: Optional[np.ndarray] = None
):
    """
    Convert protein structure to PyTorch Geometric Data object.

    Args:
        coords: CA coordinates of shape (N, 3)
        sequence: Amino acid sequence
        graph_type: 'knn' or 'contact'
        k: Number of neighbors for knn graph
        threshold: Distance threshold for contact graph
        node_features: Optional pre-computed node features

    Returns:
        torch_geometric.data.Data object
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("torch and torch_geometric are required")

    from .sequence import one_hot_encode

    # Build graph
    if graph_type == 'knn':
        edge_index, edge_attr = build_knn_graph(coords, k=k)
    else:
        edge_index, edge_attr = build_contact_graph(coords, threshold=threshold)

    # Node features (default to one-hot encoding)
    if node_features is None:
        node_features = one_hot_encode(sequence)

    # Convert to torch tensors
    x = torch.from_numpy(node_features).float()
    edge_index = torch.from_numpy(edge_index).long()
    edge_attr = torch.from_numpy(edge_attr).float().unsqueeze(-1)
    pos = torch.from_numpy(coords).float()

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        sequence=sequence,
    )


def compute_contact_map(
    coords: np.ndarray,
    threshold: float = 8.0
) -> np.ndarray:
    """
    Compute binary contact map from coordinates.

    Args:
        coords: Atom coordinates of shape (N, 3)
        threshold: Distance threshold for contacts

    Returns:
        Binary contact map of shape (N, N)
    """
    dist_matrix = compute_distance_matrix(coords)
    return (dist_matrix < threshold).astype(np.float32)
