"""
PDB file loading and structure extraction utilities.
"""

from typing import Optional, Tuple, Dict
import numpy as np

# Standard amino acid 3-letter to 1-letter mapping
AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def load_pdb(pdb_path: str, chain_id: Optional[str] = None) -> Dict:
    """
    Load a PDB file and extract structure information.

    Args:
        pdb_path: Path to PDB file
        chain_id: Specific chain to load (None for first chain)

    Returns:
        Dictionary with keys:
        - 'sequence': amino acid sequence (str)
        - 'coords': backbone coordinates N, CA, C, O (N, 4, 3)
        - 'residue_ids': residue numbers (N,)
        - 'chain_id': chain identifier
    """
    try:
        import biotite.structure.io.pdb as pdb
        import biotite.structure as struc
    except ImportError:
        raise ImportError("biotite is required: pip install biotite")

    # Load structure
    pdb_file = pdb.PDBFile.read(pdb_path)
    structure = pdb_file.get_structure(model=1)

    # Filter to protein atoms only
    structure = structure[struc.filter_amino_acids(structure)]

    # Select chain
    if chain_id is None:
        chain_id = structure.chain_id[0]
    structure = structure[structure.chain_id == chain_id]

    # Get unique residues
    residue_ids = struc.get_residues(structure)[0]

    # Extract backbone coordinates for each residue
    backbone_atoms = ['N', 'CA', 'C', 'O']
    n_residues = len(residue_ids)
    coords = np.zeros((n_residues, 4, 3), dtype=np.float32)
    sequence = []

    for i, res_id in enumerate(residue_ids):
        res_mask = structure.res_id == res_id
        res_atoms = structure[res_mask]

        # Get residue name
        res_name = res_atoms.res_name[0]
        sequence.append(AA_3TO1.get(res_name, 'X'))

        # Extract backbone atom coordinates
        for j, atom_name in enumerate(backbone_atoms):
            atom_mask = res_atoms.atom_name == atom_name
            if np.any(atom_mask):
                coords[i, j] = res_atoms.coord[atom_mask][0]

    return {
        'sequence': ''.join(sequence),
        'coords': coords,
        'residue_ids': residue_ids,
        'chain_id': chain_id,
    }


def get_backbone_coords(
    pdb_path: str,
    chain_id: Optional[str] = None,
    atom: str = 'CA'
) -> np.ndarray:
    """
    Extract specific backbone atom coordinates from a PDB file.

    Args:
        pdb_path: Path to PDB file
        chain_id: Specific chain to load
        atom: Atom type ('N', 'CA', 'C', or 'O')

    Returns:
        Coordinates array of shape (N, 3)
    """
    atom_idx = {'N': 0, 'CA': 1, 'C': 2, 'O': 3}
    if atom not in atom_idx:
        raise ValueError(f"atom must be one of {list(atom_idx.keys())}")

    data = load_pdb(pdb_path, chain_id)
    return data['coords'][:, atom_idx[atom], :]


def get_sequence_from_structure(pdb_path: str, chain_id: Optional[str] = None) -> str:
    """
    Extract amino acid sequence from a PDB file.

    Args:
        pdb_path: Path to PDB file
        chain_id: Specific chain to load

    Returns:
        Amino acid sequence string
    """
    data = load_pdb(pdb_path, chain_id)
    return data['sequence']


def download_pdb(pdb_id: str, save_path: Optional[str] = None) -> str:
    """
    Download a PDB file from RCSB.

    Args:
        pdb_id: 4-letter PDB identifier
        save_path: Path to save file (default: current directory)

    Returns:
        Path to downloaded file
    """
    try:
        from biotite.database import rcsb
    except ImportError:
        raise ImportError("biotite is required: pip install biotite")

    if save_path is None:
        save_path = '.'

    file_path = rcsb.fetch(pdb_id, 'pdb', target_path=save_path)
    return str(file_path)
