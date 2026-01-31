# Lecture 2: Protein Representations for Machine Learning

## Learning Objectives

By the end of this lecture, you will be able to:
1. Encode protein sequences using multiple representation schemes
2. Represent protein structures numerically for ML models
3. Construct graph representations of proteins
4. Understand trade-offs between different representations

## Prerequisites

- Lecture 1: Python and Data Science Basics
- Basic understanding of protein structure (primary, secondary, tertiary)

---

## The Central Challenge: Teaching Computers to See Proteins

How do you teach a computer to understand a protein?

This question lies at the heart of computational biology. When you look at a protein sequence like `MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTK...`, your brain instantly recognizes it as a string of amino acid letters. With training, a biochemist might even notice patterns—perhaps a stretch of hydrophobic residues suggesting a membrane-spanning region, or a characteristic motif hinting at enzymatic function. But a computer sees none of this. To a machine learning algorithm, that sequence is just meaningless text.

The same problem applies to protein structures. Crystallographers painstakingly determine the three-dimensional coordinates of thousands of atoms, producing beautiful images of helices and sheets folding into intricate shapes. Yet these coordinates—just lists of X, Y, Z numbers—carry no inherent meaning to an algorithm. The computer cannot "see" that two distant residues come together to form a binding pocket, or that a particular loop is flexible and functionally important.

This is where **representations** come in. A representation is simply a way of encoding information numerically so that machine learning models can process it. The choice of representation is surprisingly consequential—arguably one of the most important decisions you'll make when building a protein ML model. A good representation captures the right information in the right format, making learning easier. A poor representation can make even simple tasks impossibly hard.

Think of it this way: if you wanted to teach someone about music who had never heard a song, how would you describe it? You could write out the notes on sheet music, record the sound waves as amplitude over time, describe the chord progressions mathematically, or even try to capture the emotional arc in words. Each representation emphasizes different aspects of the music and would work better for different purposes. The same is true for proteins.

In this lecture, we'll explore the major approaches to representing proteins for machine learning. We'll start with sequences, move to structures, and finally see how graph representations unify both perspectives. Along the way, we'll understand not just *how* to implement each representation, but *why* it works and when to use it.

---

## Sequence Representations: From Letters to Numbers

Let's begin with the simplest form of protein data: the amino acid sequence. This is the primary structure—the linear chain of amino acids that constitutes the protein's genetic blueprint. Given a sequence, how do we convert it into something a neural network can work with?

### One-Hot Encoding: The Spelling Approach

The most straightforward approach is called **one-hot encoding**. Think of it like spelling out each letter of a word using a unique identifier. For proteins, we have 20 standard amino acids, so each position in the sequence becomes a 20-dimensional vector with a single "1" indicating which amino acid is present and zeros everywhere else.

Mathematically, for residue $i$ in the sequence:

$$\mathbf{x}_i \in \{0, 1\}^{20}, \quad \sum_{j=1}^{20} x_{ij} = 1$$

For example, if our amino acid alphabet is ordered as `ACDEFGHIKLMNPQRSTVWY`, then Alanine (A) would be represented as `[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`, while Cysteine (C) would be `[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`.

Let's implement this in Python. The function takes a sequence string and returns a matrix where each row corresponds to one position in the sequence.

```python
def one_hot_encode(sequence):
    """
    One-hot encode a protein sequence.

    Args:
        sequence: String of amino acids

    Returns:
        (L, 20) numpy array
    """
    aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    encoding = np.zeros((len(sequence), 20))

    for i, aa in enumerate(sequence):
        if aa in aa_to_idx:
            encoding[i, aa_to_idx[aa]] = 1.0

    return encoding
```

One-hot encoding is beautifully simple and completely interpretable. However, it has a fundamental limitation: it treats all amino acids as equally different from each other. In this representation, Alanine is just as different from Glycine (both small, hydrophobic) as it is from Tryptophan (large, aromatic). Biologically, we know this isn't true—some amino acids can substitute for each other in evolution with minimal functional impact, while other substitutions are catastrophic. One-hot encoding throws away all of this evolutionary wisdom.

### BLOSUM Encoding: Learning from Evolution

What if we could encode amino acids in a way that captures their evolutionary interchangeability? This is exactly what BLOSUM matrices provide.

**BLOSUM** stands for BLOcks SUbstitution Matrix. These matrices were derived by analyzing aligned sequences of related proteins and counting how often each pair of amino acids appears at the same position. The idea is simple but powerful: if two amino acids frequently substitute for each other across evolution, they probably have similar biochemical properties.

The BLOSUM score for amino acids $i$ and $j$ is calculated as:

$$\text{BLOSUM}_{ij} = \log_2\left(\frac{p_{ij}}{p_i \cdot p_j}\right)$$

Here, $p_{ij}$ is the observed frequency of amino acids $i$ and $j$ appearing at aligned positions, while $p_i$ and $p_j$ are their individual background frequencies. A positive score means the pair appears together more often than expected by chance—they're evolutionarily similar. A negative score means they rarely substitute—mutating one to the other would likely break the protein.

To use BLOSUM for encoding, we represent each amino acid by its entire row in the BLOSUM matrix. Now Alanine isn't just a one-hot vector; it's characterized by its substitution relationships with all other amino acids.

```python
def blosum_encode(sequence):
    """Encode using BLOSUM62 substitution scores."""
    # BLOSUM62 matrix (20x20)
    blosum62 = load_blosum62()

    aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    encoding = np.zeros((len(sequence), 20))

    for i, aa in enumerate(sequence):
        if aa in aa_to_idx:
            encoding[i] = blosum62[aa_to_idx[aa]]

    return encoding / 9.0  # Normalize to [-1, 1]
```

Why does this matter? Consider a protein binding site where a positively charged residue is essential for function. If the native amino acid is Lysine (K), a mutation to Arginine (R)—another positive residue—might be tolerated, while mutation to Glutamate (E)—a negative residue—would be catastrophic. BLOSUM captures this: K and R have a positive score (similar), while K and E have a negative score (dissimilar). When a model sees BLOSUM-encoded sequences, similar amino acids have similar representations, making it easier to learn about biochemical compatibility.

### Physicochemical Properties: Encoding What Amino Acids Do

An alternative to evolutionary encoding is to describe each amino acid by its measured biochemical properties. After all, evolution selects for function, and function emerges from chemistry.

| Property | Description |
|----------|-------------|
| Hydrophobicity | Tendency to avoid water (critical for membrane proteins and protein cores) |
| Volume | Size of the sidechain (affects packing and steric interactions) |
| Charge | +1 for Lysine/Arginine, -1 for Aspartate/Glutamate, 0 for others |
| Polarity | Capacity for hydrogen bonding |
| Aromaticity | Presence of an aromatic ring (important for pi-stacking interactions) |

Each amino acid becomes a 5-dimensional vector of these properties:

```python
PROPERTIES = {
    'A': [0.62, 0.11, 0.0, 0.0, 0.0],  # Ala: hydrophobic, small
    'K': [-1.50, 0.53, 1.0, 1.0, 0.0], # Lys: hydrophilic, positive charge
    'W': [0.81, 0.74, 0.0, 0.0, 1.0],  # Trp: mixed, aromatic
    # ... other amino acids
}

def physicochemical_encode(sequence):
    """5-dimensional property encoding."""
    encoding = np.zeros((len(sequence), 5))
    for i, aa in enumerate(sequence):
        if aa in PROPERTIES:
            encoding[i] = PROPERTIES[aa]
    return encoding
```

The beauty of physicochemical encoding is its interpretability. If a model makes a prediction, you can often understand *why* by looking at which properties drove the decision. Was the hydrophobicity pattern important? The charge distribution? This transparency is invaluable for scientific discovery.

### Learned Embeddings: Letting the Data Decide

All the encodings we've discussed so far are **hand-crafted**—designed by humans based on biological knowledge. But what if we let the data itself decide how amino acids should be represented?

This is the philosophy behind **learned embeddings**. Instead of pre-defining what each amino acid's vector should be, we initialize random vectors and let the neural network adjust them during training. The model learns whatever representation best serves the prediction task.

```python
import torch
import torch.nn as nn

class AminoAcidEmbedding(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(21, embed_dim)  # 20 AA + unknown

    def forward(self, sequence_indices):
        return self.embedding(sequence_indices)
```

The `nn.Embedding` layer maintains a trainable lookup table. Initially, each amino acid maps to a random 64-dimensional vector. As the model trains on protein data, it adjusts these vectors to be useful for prediction. After training on millions of proteins, the embeddings often reveal meaningful structure—similar amino acids cluster together in embedding space, even though no biological knowledge was explicitly provided.

This is the approach used by protein language models like **ESM** (Evolutionary Scale Modeling) from Meta AI. These models are trained on billions of protein sequences with a task analogous to language modeling: predict masked amino acids from context. The resulting embeddings capture evolutionary relationships, structural preferences, and functional properties, all learned from raw sequence data. ESM embeddings are 1280-dimensional per residue and currently represent the state-of-the-art for sequence-only protein representations.

---

## Structure Representations: Capturing Three Dimensions

While sequences tell us *what* amino acids are present, structure tells us *where* they are in space. The three-dimensional arrangement of a protein determines its function—how it binds substrates, catalyzes reactions, and interacts with partners. For structure-aware machine learning, we need representations that capture spatial relationships.

### Distance Matrices: Who Is Near Whom?

Perhaps the simplest structural representation is the **distance matrix**: a square matrix where entry $(i, j)$ gives the distance between residues $i$ and $j$. Typically, we measure distances between CA (alpha carbon) atoms, as these provide a good summary of the backbone position.

$$D_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|_2$$

```python
def compute_distance_matrix(coords):
    """
    Compute CA-CA distance matrix.

    Args:
        coords: (N, 3) CA coordinates

    Returns:
        (N, N) distance matrix
    """
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))
```

Distance matrices have elegant mathematical properties. They're symmetric ($D_{ij} = D_{ji}$), have zero diagonal ($D_{ii} = 0$), and crucially, they're **invariant to rotation and translation**. You can rotate or shift a protein in space, and its distance matrix stays exactly the same. This is important because a protein's function doesn't depend on its absolute position or orientation—only on its internal geometry.

Why does rotation invariance matter? Imagine training a model to predict protein function. If we used raw XYZ coordinates, the model might learn that "proteins with CA1 at position (10, 20, 30) have function X." But this is meaningless—the same protein could be anywhere in space! Distance matrices avoid this problem entirely.

### Contact Maps: Simplifying to Binary

A **contact map** is a binarized version of the distance matrix. Two residues are "in contact" if they're closer than some threshold (typically 8 Angstroms for CA-CA distances):

$$C_{ij} = \begin{cases} 1 & \text{if } D_{ij} < \text{threshold} \\ 0 & \text{otherwise} \end{cases}$$

```python
def compute_contact_map(coords, threshold=8.0):
    dist_matrix = compute_distance_matrix(coords)
    return (dist_matrix < threshold).astype(np.float32)
```

Contact maps throw away information about exact distances, but they highlight what's often most important: which residues interact. They're also cheaper to predict than full distance matrices, making them popular targets for early structure prediction methods.

Contacts are often categorized by sequence separation—how far apart the residues are in the linear sequence:

- **Local contacts** ($|i-j| < 6$): These arise from secondary structure. In an alpha helix, residues 4 positions apart contact due to the helical geometry.
- **Medium-range contacts** ($6 \leq |i-j| < 12$): Super-secondary motifs like helix-turn-helix.
- **Long-range contacts** ($|i-j| \geq 12$): The holy grail! These reveal tertiary structure—how distant parts of the chain come together in 3D. Predicting long-range contacts was the key breakthrough that enabled AlphaFold.

### Dihedral Angles: The Backbone's Degrees of Freedom

While distance matrices describe *pairwise* relationships, we sometimes need a *local* representation of geometry. The protein backbone has three bonds per residue, each with a **dihedral angle** (also called torsion angle) describing rotation around the bond:

- **phi (φ)**: Rotation around N-CA bond
- **psi (ψ)**: Rotation around CA-C bond
- **omega (ω)**: Rotation around C-N peptide bond (almost always ~180° due to the planar peptide bond)

These angles, rather than XYZ coordinates, are the true degrees of freedom of the backbone. Given bond lengths and angles (which are nearly constant), specifying all phi/psi values completely determines the backbone structure.

There's a technical subtlety: angles are *periodic*. An angle of 180° is the same as -180°, but numerically they're far apart. If we directly input angles to a neural network, it might think these are very different conformations! The solution is to encode each angle using sine and cosine:

```python
def encode_dihedrals(phi, psi):
    """
    Encode dihedral angles using sin/cos.

    Args:
        phi, psi: Dihedral angles in radians

    Returns:
        (N, 4) array: [sin(phi), cos(phi), sin(psi), cos(psi)]
    """
    return np.stack([
        np.sin(phi), np.cos(phi),
        np.sin(psi), np.cos(psi)
    ], axis=-1)
```

Now 180° and -180° have identical representations: $(\sin(180°), \cos(180°)) = (0, -1)$. The periodicity is handled gracefully.

Dihedral angles are particularly important for protein generation and design. Models like protein folding methods often internally work in dihedral space because generating angles directly ensures the output is a valid protein backbone (correct bond lengths and angles). In contrast, generating raw coordinates can produce distorted geometries.

### SE(3) Frames: The Language of Modern Structure Prediction

This brings us to the most sophisticated structural representation: **SE(3) frames**. This is the approach used by AlphaFold and is essential for understanding modern structure prediction.

The name comes from mathematics: SE(3) is the *Special Euclidean group in 3 dimensions*—the group of all rotations and translations in 3D space. A frame is a local coordinate system attached to each residue, consisting of:

- A **rotation matrix** $\mathbf{R}_i \in SO(3)$: How the local coordinate system is oriented relative to the global frame
- A **translation** $\mathbf{t}_i \in \mathbb{R}^3$: The position of the residue (typically the CA atom)

Together, these form a transformation:

$$T_i = (\mathbf{R}_i, \mathbf{t}_i) \in SE(3)$$

Why go to this complexity? The key insight is that proteins are made of *rigid groups*—the backbone atoms N, CA, C have fixed relative geometry within each residue. By representing each residue as a rigid body, we can describe local structure (the orientation of this residue relative to its neighbors) and global structure (the overall fold) in a unified framework.

```python
def compute_local_frame(n, ca, c):
    """
    Compute local coordinate frame for a residue.

    Args:
        n, ca, c: (3,) coordinates of backbone atoms

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation (CA position)
    """
    # X-axis: CA -> C direction
    x = c - ca
    x = x / np.linalg.norm(x)

    # Y-axis: perpendicular in NC plane
    nc = n - c
    y = nc - np.dot(nc, x) * x
    y = y / np.linalg.norm(y)

    # Z-axis: cross product
    z = np.cross(x, y)

    R = np.column_stack([x, y, z])
    t = ca

    return R, t
```

This is why AlphaFold uses frames. The structure module of AlphaFold maintains a frame for each residue and iteratively updates these frames to build the 3D structure. Operations like the Invariant Point Attention (IPA) are specifically designed to work with SE(3) frames, allowing the model to reason about protein geometry in a physically meaningful way.

---

## Graph Representations: The Best of Both Worlds

We've now seen sequence representations (1D) and structure representations (2D matrices or 3D coordinates). Is there a unified framework that elegantly handles both? Enter **graph representations**.

A graph consists of **nodes** (vertices) connected by **edges**. For proteins, the mapping is natural:
- Each residue becomes a node
- Interactions between residues become edges

This representation is powerful because it can encode both sequence (through edges connecting sequential neighbors) and structure (through edges connecting spatially close residues). Moreover, graphs can handle proteins of any length without the quadratic memory cost of full distance matrices.

### Building a K-Nearest Neighbor Graph

One common approach is to connect each residue to its $k$ nearest neighbors in 3D space:

```python
def build_knn_graph(coords, k=10):
    """
    Build k-NN graph from CA coordinates.

    Args:
        coords: (N, 3) coordinates
        k: number of neighbors

    Returns:
        edge_index: (2, E) edge list
        edge_attr: (E,) distances
    """
    dist_matrix = compute_distance_matrix(coords)

    edges_src, edges_dst, distances = [], [], []

    for i in range(len(coords)):
        dists = dist_matrix[i].copy()
        dists[i] = np.inf  # Exclude self

        neighbors = np.argsort(dists)[:k]
        for j in neighbors:
            edges_src.append(i)
            edges_dst.append(j)
            distances.append(dist_matrix[i, j])

    edge_index = np.array([edges_src, edges_dst])
    edge_attr = np.array(distances)

    return edge_index, edge_attr
```

The k-NN graph has exactly $k \times N$ edges (assuming directed edges), giving linear scaling with protein length. This is much more memory-efficient than the $N^2$ scaling of distance matrices for long proteins.

### Contact Graphs: Edges from Proximity

Alternatively, we can connect all residue pairs within a distance threshold:

```python
def build_contact_graph(coords, threshold=8.0):
    """Build graph from contact map."""
    dist_matrix = compute_distance_matrix(coords)
    contacts = (dist_matrix < threshold) & (dist_matrix > 0)

    src, dst = np.where(contacts)
    edge_index = np.array([src, dst])
    edge_attr = dist_matrix[src, dst]

    return edge_index, edge_attr
```

This creates edges based on physical interactions rather than a fixed number of neighbors.

### Enriching Nodes and Edges with Features

The power of graph representations comes from attaching rich features to nodes and edges.

**Node features** describe individual residues:
- One-hot amino acid encoding (20 dimensions)
- Physicochemical properties (5 dimensions)
- Secondary structure class (8-class: helix, sheet, coil, etc.)
- Solvent accessibility (is this residue buried or exposed?)

**Edge features** describe pairwise relationships:
- Euclidean distance (how far apart in space)
- Sequence separation (how far apart in sequence)
- Relative orientation (the rotation between local frames)
- Contact type (backbone-backbone, sidechain-sidechain, etc.)

With these features, the graph encodes a rich representation of protein chemistry and geometry.

### Putting It Together: PyTorch Geometric

For practical deep learning with graphs, PyTorch Geometric (PyG) provides a convenient data structure:

```python
from torch_geometric.data import Data

def protein_to_graph(coords, sequence, k=10):
    """Convert protein to PyG Data object."""
    # Node features
    x = torch.tensor(one_hot_encode(sequence), dtype=torch.float)

    # Edge construction
    edge_index, edge_attr = build_knn_graph(coords, k=k)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(-1)

    # Coordinates
    pos = torch.tensor(coords, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
```

This `Data` object can be directly fed into Graph Neural Networks (GNNs), which are the natural architecture for learning on graph-structured data. GNNs work by iteratively passing messages along edges, allowing each node to aggregate information from its neighbors. After several rounds of message passing, each node's representation reflects its extended neighborhood—both local sequence context and structural environment.

---

## Choosing the Right Representation

With all these options, how do you decide which representation to use? Here's a practical guide:

| Representation | Dimension | Captures | Best For |
|----------------|-----------|----------|----------|
| One-hot | L x 20 | Sequence identity | Simple baselines |
| BLOSUM | L x 20 | Evolutionary similarity | Homology detection |
| Physicochemical | L x 5 | Biochemistry | Interpretable models |
| ESM embedding | L x 1280 | Everything | State-of-the-art sequence tasks |
| Distance matrix | L x L | Pairwise structure | Structure prediction |
| Graph | Nodes + Edges | Flexible structure | GNN-based methods |
| SE(3) frames | L x 12 | Rigid body geometry | Structure prediction/generation |

### Rules of Thumb

1. **Sequence-only tasks (no structure available)**: Start with ESM embeddings. They encode evolutionary and structural information learned from billions of sequences. If you need interpretability or have limited compute, physicochemical or BLOSUM encodings are reasonable alternatives.

2. **Structure prediction**: Distance matrices or contact maps are natural prediction targets. Modern methods predict these first, then reconstruct 3D coordinates.

3. **Structure-based tasks (structure is input)**: Graph representations shine here. They're efficient, flexible, and work naturally with GNNs.

4. **Generative models**: Dihedral angles or SE(3) frames. These representations respect the physical constraints of protein geometry, making it easier to generate valid structures.

---

## Practical Considerations

### Memory Matters

For a protein of length $L$:
- One-hot encoding: $O(L \times 20)$ — linear
- Distance matrix: $O(L^2)$ — quadratic
- Graph (k-NN): $O(L \times k)$ — linear

This difference becomes critical for long proteins. A 1000-residue protein's distance matrix has a million entries; a 2000-residue protein has four million. Graph representations with fixed $k$ scale linearly, making them essential for very long proteins or large batch sizes.

### Handling Variable Lengths

Proteins come in all sizes. When batching multiple proteins for training, you have two main options:

```python
def collate_proteins(batch):
    """Collate proteins of different lengths."""
    # Option 1: Pad to max length in batch
    max_len = max(p['length'] for p in batch)
    # ... pad sequences and masks ...

    # Option 2: Use PyG's graph batching
    from torch_geometric.data import Batch
    return Batch.from_data_list([p['graph'] for p in batch])
```

PyG's graph batching is elegant—it combines multiple graphs into one large graph with disconnected components, avoiding the need for padding.

### Normalization

Neural networks train better when input features have similar magnitudes. Always normalize:

- **Z-score normalization**: $(x - \mu) / \sigma$ — centers features around zero with unit variance
- **Min-max normalization**: $(x - \min) / (\max - \min)$ — scales to [0, 1]
- **Angles**: Always use sin/cos encoding to handle periodicity

---

## Key Takeaways

1. **Representations are the foundation of protein ML.** The choice of representation shapes what information your model can access and how easily it can learn.

2. **There's no single best representation.** One-hot is simple, BLOSUM encodes evolution, physicochemical properties aid interpretation, and learned embeddings (ESM) often achieve the best performance.

3. **Structure representations must handle geometry carefully.** Distance matrices provide rotation invariance, dihedral angles encode local geometry compactly, and SE(3) frames provide the rigorous foundation for modern structure prediction.

4. **Graphs unify sequence and structure.** They're efficient, flexible, and naturally suited to GNN architectures.

5. **Consider computational constraints.** Long proteins may require sparse representations like graphs rather than dense distance matrices.

---

## Exercises

1. Implement one-hot, BLOSUM, and physicochemical encodings for the same protein. Plot the encoded vectors—can you see amino acid similarities in the BLOSUM encoding?

2. Download ubiquitin (PDB: 1UBQ) and build a k-NN graph with k=10. Visualize the graph—do the edges correspond to contacts you'd expect from the structure?

3. Extract phi/psi angles from a protein and create a Ramachandran plot. Color by secondary structure—can you see the clusters?

4. Create a PyTorch Geometric dataset from a set of PDB files. Train a simple GNN to predict secondary structure from graph features.

---

## References

- Henikoff & Henikoff (1992). "Amino acid substitution matrices from protein blocks" — The original BLOSUM paper
- AlQuraishi (2019). "End-to-End Differentiable Learning of Protein Structure" — Pioneering work on dihedral representations
- Jumper et al. (2021). "Highly accurate protein structure prediction with AlphaFold" — SE(3) frames and IPA
- Rives et al. (2021). "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences" — ESM embeddings
