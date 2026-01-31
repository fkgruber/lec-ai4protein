# Lecture 1: Python and Data Science Basics for Protein AI

## Introduction: Why Data Handling is the Foundation of Protein AI

Imagine you are handed a dataset containing millions of protein sequences, each one a string of letters representing the amino acid building blocks of life. Your goal is to train a neural network that can predict whether a new protein will fold correctly, bind to a drug molecule, or cause disease when mutated. Before you can build any sophisticated deep learning model, you face a fundamental question: *how do you even represent a protein in a way that a computer can understand?*

This question sits at the heart of computational biology, and it is why mastering data handling is the essential first step in your journey into protein AI. Proteins are not simple data points like house prices or customer reviews. They are complex three-dimensional machines with variable lengths, intricate geometries, and evolutionary relationships that span billions of years. The way you load, transform, and prepare protein data will determine whether your models succeed or fail.

In this lecture, we will build the foundational toolkit you need to work with protein data in Python. We will explore NumPy for numerical computations, Pandas for tabular data analysis, and specialized libraries for reading protein file formats. More importantly, we will understand *why* each of these tools matters in the context of protein AI. By the end, you will be equipped to take raw biological data and transform it into the numerical representations that power modern machine learning models.

---

## Learning Objectives

By the end of this lecture, you will be able to:
1. Perform efficient numerical computations using NumPy arrays
2. Manipulate and analyze tabular protein data with Pandas
3. Load and parse common protein file formats (FASTA for sequences, PDB for structures)
4. Prepare biological data for machine learning pipelines while avoiding common pitfalls

## Prerequisites

- Basic Python programming (variables, functions, loops)
- Familiarity with command line operations

---

## 1. NumPy: The Language of Numerical Protein Data

When biologists think about proteins, they think about sequences of amino acids and complex three-dimensional shapes. But when computers process proteins, they think in numbers. Every protein sequence becomes a matrix of values. Every atomic position becomes a triplet of coordinates. Every evolutionary relationship becomes a numerical distance. NumPy, the Numerical Python library, is the translator that bridges these two worlds.

Why is NumPy so essential? Consider a simple task: comparing two protein structures to see how similar they are. This requires computing the distance between every pair of corresponding atoms, potentially thousands of calculations. If you wrote this as nested Python loops, it would take seconds or even minutes. With NumPy, the same calculation completes in milliseconds. This speed advantage becomes critical when you are training deep learning models that process millions of protein examples.

### 1.1 Representing Proteins as Arrays

The fundamental data structure in NumPy is the **array**, a grid of values that can have any number of dimensions. For protein AI, we use arrays to represent three main types of data:

- **Sequence encodings**: A protein sequence like "MVLSPADKTN..." becomes a matrix where each row represents a residue position and each column represents one of the 20 standard amino acids.
- **3D coordinates**: Every atom in a protein structure has an (x, y, z) position in space, naturally represented as an N-by-3 array where N is the number of atoms.
- **Contact maps and distance matrices**: These N-by-N matrices capture which residues are close together in the folded structure, providing a simplified view of protein topology.

Let us create our first protein-related arrays:

```python
import numpy as np

# Creating arrays
sequence_encoding = np.zeros((100, 20))  # 100 residues, 20 amino acids
coordinates = np.random.randn(76, 3)      # 76 CA atoms, xyz

# Key properties
print(f"Shape: {coordinates.shape}")     # (76, 3)
print(f"Dtype: {coordinates.dtype}")     # float64
print(f"Size: {coordinates.size}")       # 228
```

The `shape` tells us the dimensions of our array, which for coordinates means 76 atoms with 3 coordinates each. The `dtype` specifies the numerical precision (float64 means 64-bit floating point numbers, offering about 15 decimal digits of precision). The `size` is simply the total number of elements. Understanding these properties is crucial for debugging and memory management when working with large protein datasets.

### 1.2 Broadcasting: Elegant Operations Across Dimensions

One of NumPy's most powerful features is **broadcasting**, which allows operations between arrays of different shapes by automatically expanding the smaller array to match the larger one. This might sound abstract, but it solves a very concrete problem in protein structure analysis.

Consider centering a protein structure, a common preprocessing step that moves the center of mass to the origin. Without broadcasting, you would need to write a loop that subtracts the centroid from each atom individually. With broadcasting, a single line suffices:

```python
# Center coordinates (subtract mean from each column)
coords = np.random.randn(100, 3)
centroid = coords.mean(axis=0)  # Shape: (3,)
centered = coords - centroid    # Broadcasting: (100, 3) - (3,) -> (100, 3)
```

What happens here? The centroid is a 1D array with 3 values (the mean x, mean y, and mean z). When we subtract it from coords, NumPy automatically broadcasts the centroid across all 100 rows. Each atom's coordinates get the same centroid subtracted. This operation is not only elegant but also runs at C-level speed, far faster than any Python loop.

Broadcasting follows a simple rule: dimensions are compatible when they are equal or when one of them is 1. NumPy automatically expands dimensions of size 1 to match the other array. This rule will become second nature as you work through more examples.

### 1.3 Distance Matrices: The Fingerprint of Protein Structure

If there is one operation that defines structural biology, it is computing distances between atoms. A **distance matrix** captures the pairwise distances between all atoms in a structure, creating an N-by-N symmetric matrix that serves as a fingerprint of the protein's fold.

Why are distance matrices so important? Because they encode the essential information about protein structure in a rotation- and translation-invariant form. Two proteins with identical distance matrices have identical structures, regardless of how they are oriented in space. This property makes distance matrices invaluable for structure comparison, contact prediction, and as input features for neural networks.

The mathematical operation is straightforward: for each pair of atoms i and j, compute the Euclidean distance. However, implementing this efficiently requires a clever use of broadcasting:

```python
def compute_distance_matrix(coords):
    """
    Compute pairwise distances between all atoms.

    Args:
        coords: (N, 3) array of atom coordinates

    Returns:
        (N, N) distance matrix
    """
    # Using broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))
```

Let us unpack this line by line. The expression `coords[:, np.newaxis, :]` reshapes our (N, 3) array to (N, 1, 3) by inserting a new axis. Similarly, `coords[np.newaxis, :, :]` creates a (1, N, 3) array. When we subtract these, broadcasting expands both to (N, N, 3), giving us the coordinate differences for every pair of atoms. The final steps square these differences, sum along the coordinate axis, and take the square root to get Euclidean distances.

This single function will appear throughout your protein AI journey. It is used for contact map prediction, structure validation, and as a key input to geometric deep learning models like AlphaFold.

### 1.4 Linear Algebra: The Mathematics of Structure Alignment

Beyond basic arithmetic, protein structure analysis relies heavily on linear algebra operations like matrix multiplication and singular value decomposition (SVD). These operations enable structure alignment, a fundamental task where we superimpose two protein structures to measure their similarity.

The **Kabsch algorithm** is the gold standard for finding the optimal rotation to align two sets of corresponding points. It uses SVD to decompose the covariance matrix between the two structures and extract the rotation that minimizes root-mean-square deviation (RMSD). Understanding this algorithm is essential for evaluating structure prediction models:

```python
# Kabsch algorithm for structure superposition
def kabsch_rotation(mobile, target):
    """Find optimal rotation matrix to align mobile to target."""
    H = mobile.T @ target
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R
```

The SVD decomposes the covariance matrix H into three components. The rotation matrix R combines the U and V matrices to find the optimal alignment. The reflection check ensures we get a proper rotation (not a mirror flip), which is crucial for biological correctness.

**Key insight**: NumPy provides the building blocks for representing proteins numerically and performing the geometric calculations central to structural biology. Mastering array operations, broadcasting, and linear algebra will serve you throughout this course and your career in computational biology.

---

## 2. Pandas: Wrangling Tabular Protein Data

While NumPy excels at numerical arrays, real-world protein datasets come in messier formats: CSV files from experiments, database dumps with mixed data types, annotation tables with missing values. This is where Pandas shines. Pandas provides the **DataFrame**, a two-dimensional table with labeled rows and columns that can hold different data types and handle missing values gracefully.

Protein datasets often combine numerical features (molecular weight, binding affinity) with categorical data (organism, subcellular location) and text (sequences, annotations). Pandas lets you filter, aggregate, and transform all of these in a unified framework. Whether you are analyzing protein solubility data, curating a training set, or computing summary statistics, Pandas will be your primary tool.

### 2.1 Loading and Exploring Protein Datasets

Most protein machine learning datasets come as CSV files with sequences and labels. The DeepSol dataset, for example, contains protein sequences labeled as soluble (1) or insoluble (0), a classic binary classification task:

```python
import pandas as pd

# Example: DeepSol solubility dataset
df = pd.read_csv('deepsol_train.csv')
print(df.head())
#    sequence                                            label
# 0  MKWVTFISLLFLFSSAYSRGVFRRDTHKSEIAHRFKDLGE...        1
# 1  MDPKISEMHPALRLVDPQIQLAVTRILDPDGNVLDKARKV...        0
```

The `head()` method shows the first few rows, giving you a quick sense of the data structure. But for protein datasets, you often want more: How many examples do you have? Are the classes balanced? How long are the sequences?

```python
# Basic statistics
print(f"Dataset size: {len(df)}")
print(f"Class distribution:\n{df['label'].value_counts()}")

# Sequence length analysis
df['length'] = df['sequence'].str.len()
print(f"Length range: {df['length'].min()} - {df['length'].max()}")
print(f"Mean length: {df['length'].mean():.1f}")
```

Sequence length analysis is particularly important for proteins. Unlike images that can be resized to a fixed dimension, proteins have intrinsic lengths ranging from tens to thousands of amino acids. This variability affects model architecture choices and computational requirements.

### 2.2 Filtering and Selecting Subsets

Pandas excels at selecting subsets of data based on conditions. This is essential for creating focused training sets, analyzing specific protein families, or excluding problematic entries:

```python
# Filter by length
short_proteins = df[df['length'] < 200]

# Select soluble proteins
soluble = df[df['label'] == 1]

# Complex queries
filtered = df[(df['length'] >= 50) & (df['length'] <= 500) & (df['label'] == 1)]
```

The syntax `df[condition]` returns rows where the condition is True. For compound conditions, use `&` (and), `|` (or), and wrap each condition in parentheses. This filtering capability is invaluable for creating clean, well-defined training sets.

### 2.3 Feature Engineering: From Sequences to Numbers

Raw protein sequences are strings of letters, but machine learning models need numbers. **Feature engineering** transforms sequences into numerical representations that capture biologically meaningful properties. The simplest approach is amino acid composition: the fraction of each amino acid type in the sequence.

```python
# Amino acid composition
def aa_composition(sequence):
    """Compute amino acid frequencies."""
    from collections import Counter
    counts = Counter(sequence)
    total = len(sequence)
    return {aa: counts.get(aa, 0) / total for aa in 'ACDEFGHIKLMNPQRSTVWY'}

# Apply to all sequences
compositions = df['sequence'].apply(aa_composition)
aa_df = pd.DataFrame(compositions.tolist())
df = pd.concat([df, aa_df], axis=1)
```

This creates 20 new columns, one for each standard amino acid, containing the fraction of that amino acid in each protein. While simple, amino acid composition captures important properties: proteins rich in charged residues (K, R, E, D) tend to be soluble, while those heavy in hydrophobic residues (V, I, L, F) may aggregate.

**Key insight**: Pandas transforms the messy reality of biological data into clean, structured tables ready for analysis and machine learning. Learn to leverage its filtering, grouping, and transformation capabilities to extract maximum value from your protein datasets.

---

## 3. Protein File Formats: The Languages of Structural Biology

The computational biology community has developed standardized file formats for storing protein data. Understanding these formats is essential because they are the raw materials from which you build datasets. The two most important formats are FASTA for sequences and PDB for three-dimensional structures.

### 3.1 FASTA: The Universal Sequence Format

FASTA is beautifully simple: a header line starting with `>` followed by one or more lines of sequence. The header typically contains identifiers and annotations, while the sequence uses single-letter amino acid codes:

```
>sp|P0A6Y8|DNAK_ECOLI Chaperone protein DnaK
MGKIIGIDLGTTNSCVAIMDGTTPRVLENAEGDRTTPSIIAYTQDGETLVGQPAKRQAVT
NPQNTLFAIKRLIGRRFQDEEVQRDVSIMPFKIIAADNGDAWVEVKGQKMAPPQISAEVL
...
```

The header in this example follows UniProt conventions: `sp` indicates Swiss-Prot (curated database), `P0A6Y8` is the accession number, `DNAK_ECOLI` is the entry name, and the rest describes the protein. Different databases use different header conventions, so always check the source.

### 3.2 Parsing FASTA with Biopython

Biopython is the standard library for biological file parsing in Python. Its `SeqIO` module handles many sequence formats with a consistent interface:

```python
from Bio import SeqIO

def load_fasta(filepath):
    """Load sequences from FASTA file."""
    sequences = {}
    for record in SeqIO.parse(filepath, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

# Load and examine
seqs = load_fasta("proteins.fasta")
for name, seq in list(seqs.items())[:3]:
    print(f"{name}: {len(seq)} residues")
```

The `SeqIO.parse()` function returns an iterator of SeqRecord objects, each containing the sequence and metadata. This design handles files of any size efficiently, reading one record at a time rather than loading everything into memory.

### 3.3 PDB: The Format for 3D Structures

While FASTA captures the *sequence* of a protein, the **PDB format** captures its *structure*: the three-dimensional positions of every atom. PDB files are fixed-width text files with a specific column layout that dates back to the punch card era:

```
ATOM      1  N   MET A   1      27.340  24.430   2.614  1.00  9.67           N
ATOM      2  CA  MET A   1      26.266  25.413   2.842  1.00 10.38           C
ATOM      3  C   MET A   1      26.913  26.639   3.531  1.00  9.62           C
...
```

Understanding the column layout is essential for writing custom parsers:

- **Columns 1-6**: Record type (ATOM for coordinates)
- **Columns 7-11**: Atom serial number
- **Columns 13-16**: Atom name (N, CA, C, O for backbone; varies for side chains)
- **Columns 18-20**: Residue name (three-letter code like MET, ALA)
- **Column 22**: Chain identifier
- **Columns 23-26**: Residue sequence number
- **Columns 31-54**: X, Y, Z coordinates in Angstroms

The **CA** (alpha carbon) atoms are particularly important. Every amino acid has exactly one CA, located at the backbone's central carbon. The CA trace provides a simplified representation of protein structure that is widely used in machine learning.

### 3.4 Parsing PDB with Biotite

Biotite is a modern alternative to Biopython for structural biology, with clean APIs and good performance. Here is how to load a PDB file and extract CA coordinates:

```python
import biotite.structure.io.pdb as pdb
import biotite.structure as struc

def load_pdb(filepath, chain='A'):
    """Load structure from PDB file."""
    pdb_file = pdb.PDBFile.read(filepath)
    structure = pdb_file.get_structure(model=1)

    # Filter to protein atoms only
    structure = structure[struc.filter_amino_acids(structure)]

    # Select chain
    if chain:
        structure = structure[structure.chain_id == chain]

    return structure

# Extract CA coordinates
structure = load_pdb("1ubq.pdb")
ca_mask = structure.atom_name == "CA"
ca_coords = structure.coord[ca_mask]
print(f"CA coordinates shape: {ca_coords.shape}")  # (76, 3)
```

Ubiquitin (PDB ID: 1UBQ) is a small protein with 76 residues, so we get 76 CA coordinates. These coordinates can now be fed into any of our NumPy functions: compute the distance matrix, center the structure, or align it to another protein.

### 3.5 Bridging Sequence and Structure

Sometimes you need to extract the sequence from a structure file, perhaps to verify consistency or because you only have the PDB. This requires mapping three-letter residue codes to single letters:

```python
AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def get_sequence(structure):
    """Extract amino acid sequence from structure."""
    residue_ids, residue_names = struc.get_residues(structure)
    sequence = ''.join(AA_3TO1.get(name, 'X') for name in residue_names)
    return sequence
```

Non-standard amino acids (modified residues, selenomethionine for crystallography) are mapped to 'X', the unknown residue code. Real-world PDB files often contain such surprises, so always validate your parsing.

**Key insight**: FASTA and PDB are the raw materials of protein AI. Mastering their parsing with Biopython and Biotite gives you access to the wealth of publicly available protein data.

---

## 4. Preparing Data for Machine Learning

You have loaded your sequences and structures, computed features, and explored your dataset. Now comes the critical step: preparing data for machine learning. This stage is where many practitioners make subtle errors that doom their models to overfit and fail on real-world data.

### 4.1 Train/Validation/Test Splits

The standard approach divides data into three sets: training (to fit the model), validation (to tune hyperparameters), and test (to evaluate final performance). A typical 80/10/10 split works for many applications:

```python
from sklearn.model_selection import train_test_split

# Standard 80/10/10 split
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

The `random_state` ensures reproducibility. Always set this so that you and your collaborators get the same splits.

### 4.2 The Data Leakage Problem in Protein ML

Here is where protein data differs fundamentally from images or text: **sequence similarity causes data leakage**. If your training set contains a protein 90% identical to one in your test set, your model can achieve high test accuracy simply by memorizing the training example, without learning anything generalizable.

This is not a theoretical concern. Many published protein prediction methods have been later shown to overestimate performance due to data leakage. To build truly predictive models, you must split data based on:

- **Sequence identity clusters**: Group proteins sharing >30% identity, then split at the group level
- **Protein families**: Use CATH or SCOP classifications to ensure different folds are separated
- **Time**: Train on structures solved before a cutoff date, test on newer ones (mimics real deployment)

```python
# Example: Split by CATH superfamily
def split_by_family(df, family_col='superfamily', test_frac=0.2):
    families = df[family_col].unique()
    np.random.shuffle(families)
    n_test = int(len(families) * test_frac)

    test_families = set(families[:n_test])
    train_mask = ~df[family_col].isin(test_families)

    return df[train_mask], df[~train_mask]
```

This function ensures that no protein family appears in both train and test sets, providing a much more realistic estimate of model performance on novel proteins.

### 4.3 Handling Variable-Length Sequences

Unlike images that can be resized to 224x224 pixels, proteins have intrinsic lengths. A 50-residue peptide and a 500-residue enzyme cannot simply be stretched to the same size. You have several options:

1. **Padding**: Add zeros (or a special token) to make all sequences the same length
2. **Truncation**: Cut sequences to a maximum length
3. **Bucketing**: Group sequences by similar length to minimize padding

Padding is most common for simple models:

```python
def pad_sequences(sequences, max_len=None, pad_value=0):
    """Pad sequences to same length."""
    if max_len is None:
        max_len = max(len(s) for s in sequences)

    padded = np.full((len(sequences), max_len), pad_value)
    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]

    return padded
```

Modern deep learning frameworks have sophisticated mechanisms for masking padded positions so they do not affect training. We will explore these in later lectures on transformer architectures.

**Key insight**: Data preparation for protein ML requires careful attention to data leakage and variable-length handling. Proper splitting based on sequence similarity is essential for realistic performance evaluation.

---

## 5. Visualizing Protein Data

The human visual system is a powerful pattern-recognition engine. Effective visualizations help you understand your data, catch errors, and communicate results. For proteins, two visualizations are particularly important: contact maps that show which residues are close together, and 3D structure views that display the actual fold.

### 5.1 Contact Maps: A 2D View of 3D Structure

A **contact map** is a binary matrix where position (i, j) is True if residues i and j are within some distance threshold (typically 8 Angstroms between CA atoms). Contact maps encode the essential topology of a protein fold in a simple 2D image:

```python
import matplotlib.pyplot as plt

def plot_contact_map(coords, threshold=8.0):
    """Plot residue contact map."""
    dist_matrix = compute_distance_matrix(coords)
    contacts = dist_matrix < threshold

    plt.figure(figsize=(8, 8))
    plt.imshow(contacts, cmap='Blues', origin='lower')
    plt.xlabel('Residue Index')
    plt.ylabel('Residue Index')
    plt.title(f'Contact Map (threshold={threshold}A)')
    plt.colorbar(label='Contact')
    plt.show()
```

Contact maps reveal secondary structure elements: alpha helices appear as thick diagonals (consecutive residues in contact), while beta sheets create off-diagonal stripes (distant residues brought together by hydrogen bonding). Contact prediction from sequence alone was one of the first successful applications of deep learning to protein structure.

### 5.2 Interactive 3D Structure Visualization

For true structural insight, nothing beats a 3D view. The py3Dmol library brings interactive molecular visualization to Jupyter notebooks:

```python
import py3Dmol

def view_structure(pdb_path):
    """Interactive 3D structure viewer."""
    with open(pdb_path) as f:
        pdb_string = f.read()

    viewer = py3Dmol.view(width=600, height=400)
    viewer.addModel(pdb_string, 'pdb')
    viewer.setStyle({'cartoon': {'color': 'spectrum'}})
    viewer.zoomTo()
    return viewer
```

The 'spectrum' coloring runs from blue at the N-terminus to red at the C-terminus, helping you follow the protein chain. You can rotate, zoom, and explore the structure interactively, building intuition about protein architecture.

**Key insight**: Visualization is not just for presentation, it is a critical tool for understanding your data and catching preprocessing errors before they propagate to your models.

---

## Key Takeaways

1. **NumPy** provides the numerical foundation for protein AI. Broadcasting enables elegant vectorized operations, and distance matrices are the fundamental representation of protein structure.

2. **Pandas** handles the messy reality of biological datasets. Use it for loading data, filtering subsets, and engineering features from raw sequences.

3. **FASTA** stores sequences, **PDB** stores structures. Biopython and Biotite are your parsing libraries of choice. Always validate parsed data against expectations.

4. **Data splitting** in protein ML must account for sequence similarity. Random splits cause data leakage and lead to overestimated performance. Split by sequence clusters or protein families.

5. **Visualization** with contact maps and 3D viewers helps you understand protein structure and catch errors early.

These tools and techniques form the bedrock upon which we will build increasingly sophisticated protein AI methods. In the next lecture, we will explore how to represent proteins for deep learning, moving from hand-crafted features to learned embeddings.

---

## Exercises

1. **Amino acid analysis**: Load a sample UniProt FASTA file and compute amino acid composition for all sequences. Which amino acids are most common? Least common?

2. **Contact analysis**: Calculate the distance matrix for ubiquitin (1UBQ) and identify all residue pairs within 5 Angstroms. How many contacts does each residue make on average?

3. **Sequence-identity splitting**: Create a train/test split of a protein dataset where no two proteins in different splits share >30% sequence identity. You can use CD-HIT or MMseqs2 for clustering, then split at the cluster level.

---

## References

- [NumPy Documentation](https://numpy.org/doc/) - The definitive reference for array operations
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/) - Comprehensive tutorial for data manipulation
- [Biopython Tutorial](https://biopython.org/DIST/docs/tutorial/Tutorial.html) - Standard library for biological sequences
- [Biotite Documentation](https://www.biotite-python.org/) - Modern library for structural biology
- [PDB File Format](https://www.wwpdb.org/documentation/file-format) - Official specification from the Protein Data Bank
