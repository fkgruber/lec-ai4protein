"""
Visualization utilities for protein data.
"""

from typing import Optional, List, Union
import numpy as np


def plot_contact_map(
    contact_map: np.ndarray,
    title: str = "Contact Map",
    cmap: str = "Blues",
    figsize: tuple = (8, 8),
    ax=None
):
    """
    Plot a protein contact map.

    Args:
        contact_map: Contact map array of shape (N, N)
        title: Plot title
        cmap: Matplotlib colormap
        figsize: Figure size
        ax: Optional matplotlib axis

    Returns:
        Matplotlib axis
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(contact_map, cmap=cmap, origin='lower')
    ax.set_xlabel('Residue Index')
    ax.set_ylabel('Residue Index')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return ax


def plot_attention_map(
    attention: np.ndarray,
    title: str = "Attention Map",
    cmap: str = "viridis",
    figsize: tuple = (10, 8),
    head_idx: Optional[int] = None
):
    """
    Plot attention weights from a transformer layer.

    Args:
        attention: Attention weights of shape (heads, L, L) or (L, L)
        title: Plot title
        cmap: Matplotlib colormap
        figsize: Figure size
        head_idx: Specific attention head to plot (None for average)

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    if attention.ndim == 3:
        n_heads = attention.shape[0]
        if head_idx is not None:
            attention = attention[head_idx]
            title = f"{title} (Head {head_idx})"
        else:
            attention = attention.mean(axis=0)
            title = f"{title} (Average over {n_heads} heads)"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attention, cmap=cmap)
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    return fig


def view_structure_3d(
    coords: np.ndarray,
    sequence: Optional[str] = None,
    color_by: str = 'residue',
    style: str = 'cartoon',
    width: int = 600,
    height: int = 400
):
    """
    View protein structure in 3D using py3Dmol.

    Args:
        coords: Backbone coordinates (N, 4, 3) or CA only (N, 3)
        sequence: Amino acid sequence for coloring
        color_by: 'residue', 'chain', or 'rainbow'
        style: 'cartoon', 'stick', 'sphere', or 'line'
        width: Viewer width
        height: Viewer height

    Returns:
        py3Dmol viewer object
    """
    try:
        import py3Dmol
    except ImportError:
        raise ImportError("py3Dmol is required: pip install py3Dmol")

    # Generate PDB string from coordinates
    pdb_lines = []

    if coords.ndim == 2:
        # CA only
        for i, coord in enumerate(coords):
            res_name = 'ALA' if sequence is None else _aa_1to3(sequence[i])
            line = f"ATOM  {i*4+2:5d}  CA  {res_name} A{i+1:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           C"
            pdb_lines.append(line)
    else:
        # Full backbone (N, CA, C, O)
        atom_names = ['N', 'CA', 'C', 'O']
        atom_elements = ['N', 'C', 'C', 'O']
        atom_num = 1
        for i, res_coords in enumerate(coords):
            res_name = 'ALA' if sequence is None else _aa_1to3(sequence[i])
            for j, (atom, elem) in enumerate(zip(atom_names, atom_elements)):
                coord = res_coords[j]
                line = f"ATOM  {atom_num:5d}  {atom:3s} {res_name} A{i+1:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           {elem}"
                pdb_lines.append(line)
                atom_num += 1

    pdb_lines.append("END")
    pdb_string = '\n'.join(pdb_lines)

    # Create viewer
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(pdb_string, 'pdb')

    # Set style and color
    if color_by == 'rainbow':
        viewer.setStyle({style: {'color': 'spectrum'}})
    elif color_by == 'residue':
        viewer.setStyle({style: {'colorscheme': 'amino'}})
    else:
        viewer.setStyle({style: {}})

    viewer.zoomTo()
    return viewer


def _aa_1to3(aa: str) -> str:
    """Convert 1-letter amino acid code to 3-letter."""
    mapping = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
        'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
        'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
        'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
    }
    return mapping.get(aa.upper(), 'ALA')


def plot_loss_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Progress",
    figsize: tuple = (10, 6)
):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: Optional list of validation losses per epoch
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)

    if val_losses is not None:
        ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 8),
    normalize: bool = True
):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class names
        title: Plot title
        figsize: Figure size
        normalize: Whether to normalize by row

    Returns:
        Matplotlib figure
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    if labels is not None:
        ax.set(xticks=np.arange(len(labels)),
               yticks=np.arange(len(labels)),
               xticklabels=labels,
               yticklabels=labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    return fig
