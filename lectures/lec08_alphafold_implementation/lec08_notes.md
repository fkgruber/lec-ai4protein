# Lecture 8: AlphaFold Implementation Case Study

## Learning Objectives

By the end of this lecture, you will be able to:
1. Understand the end-to-end architecture of AlphaFold2
2. Explain how Evoformer processes MSA and pair representations
3. Describe the Structure Module's invariant point attention mechanism
4. Implement simplified versions of key AlphaFold2 components

---

## Introduction: The Fifty-Year Challenge

In 1972, Christian Anfinsen stood before the Royal Swedish Academy of Sciences to receive his Nobel Prize in Chemistry. His acceptance speech crystallized a hypothesis that would haunt structural biologists for half a century: a protein's amino acid sequence contains all the information necessary to determine its three-dimensional structure. The sequence, Anfinsen argued, dictates the fold.

This was simultaneously obvious and impossibly cryptic. Obvious because proteins do fold, reliably and reproducibly, millions of times per second in every living cell. Cryptic because despite knowing that the answer was encoded in the sequence, no one could read the code. A protein of just 100 amino acids can, in principle, adopt an astronomical number of conformations. If a protein tried a trillion different configurations per second, it would take longer than the age of the universe to find the right one by random search. Yet real proteins fold in milliseconds.

This paradox, known as Levinthal's paradox, suggested that evolution had found shortcuts, that the folding landscape must be funneled rather than flat. But understanding that shortcuts existed was different from knowing what they were. For decades, predicting protein structure from sequence remained the grand challenge of computational biology.

Then, in November 2020, everything changed.

DeepMind's AlphaFold2 achieved what many had thought impossible within their lifetimes. At the Critical Assessment of protein Structure Prediction (CASP14) competition, it predicted protein structures with accuracy approaching experimental methods. A problem that had resisted fifty years of effort seemed to yield almost overnight to deep learning.

But AlphaFold2 was not simply a bigger neural network thrown at the problem. Its architecture embodied deep biological insight, translating decades of accumulated knowledge about protein evolution and geometry into the language of attention mechanisms and learned representations. Understanding how AlphaFold2 works is not just an exercise in neural network archaeology; it reveals fundamental principles about how proteins encode structural information and how that information can be extracted through computation.

This lecture walks through AlphaFold2's architecture piece by piece, explaining not just what each component does but why it was designed that way. We will see how evolutionary information flows through the network, how geometric constraints are enforced, and how three-dimensional structure emerges from one-dimensional sequence.

---

## 1. The Bird's Eye View: How AlphaFold2 Thinks About Proteins

Before diving into code and equations, let us develop intuition for AlphaFold2's overall strategy.

### The Core Insight: Proteins Are Not Alone

When you try to predict the structure of a protein, you might think you are working with limited information, just a string of amino acid letters. But here is the crucial insight: no protein exists in isolation. Every protein has relatives, sequences that diverged from a common ancestor millions of years ago and have been independently refined by natural selection ever since.

These related sequences form what is called a Multiple Sequence Alignment, or MSA. In an MSA, related sequences are aligned so that evolutionarily equivalent positions line up in columns. When you examine an MSA, you discover something remarkable: certain positions vary freely while others are rigidly conserved. The conserved positions often turn out to be structurally or functionally critical, places where mutations would break the protein.

But the really valuable information lies not in which positions are conserved, but in which positions co-vary. If you find that when position 15 is valine, position 47 tends to be leucine, and when position 15 mutates to isoleucine, position 47 compensates by becoming methionine, you have discovered a correlated mutation. These correlations whisper secrets about structure: positions that co-evolve are often in physical contact in the folded protein. Change one, and you must change its partner to maintain the interaction.

AlphaFold2's first major insight was to make evolutionary information not just an input feature but the central organizing principle of its architecture. The network processes MSA information through specialized attention mechanisms designed to extract co-evolutionary signals and translate them into structural predictions.

### Two Representations, One Structure

AlphaFold2 maintains two parallel representations throughout most of its computation:

**The MSA representation** captures information about each position in each sequence of the alignment. If you have 512 related sequences aligned over 200 residue positions, your MSA representation is a 512 by 200 grid, where each cell knows about that particular position in that particular sequence. This representation learns which positions are coupled across evolution.

**The pair representation** captures relationships between pairs of positions in the target sequence. For a 200-residue protein, this is a 200 by 200 matrix where cell (i, j) encodes what the network believes about the relationship between residue i and residue j. Are they close in space? Do they form a hydrogen bond? Are they part of the same secondary structure element?

These two representations talk to each other throughout the network, with evolutionary information from the MSA informing pairwise relationships, and pairwise constraints helping interpret the MSA. By the end of this information exchange, the pair representation contains a detailed map of which residues are near each other in space, essentially a blurry picture of the contact map.

### From Distances to Coordinates

Knowing that residue 15 is close to residue 47 is useful, but it does not directly give you three-dimensional coordinates. The final component of AlphaFold2, the Structure Module, takes the refined representations and converts them into actual atomic positions.

This conversion is subtle because protein structures live in three-dimensional Euclidean space, which has symmetries that the network must respect. If you rotate a protein, it is still the same protein. If you translate it across the room, the internal structure is unchanged. The Structure Module uses a specialized attention mechanism called Invariant Point Attention that respects these symmetries, ensuring that the network learns geometric relationships rather than arbitrary coordinate systems.

Let us now examine each component in detail.

---

## 2. Input Embedding: Translating Biology into Tensors

Every deep learning system must bridge the gap between its domain and the world of tensors and gradients. For AlphaFold2, this means encoding amino acid sequences, evolutionary alignments, and structural templates into numerical representations the network can process.

### What Goes In

AlphaFold2 accepts several inputs:

The **target sequence** is the protein whose structure you want to predict, represented as a string of amino acid identifiers. A sequence like "MVLSPADKTN..." gets converted to numerical indices (methionine is 0, valine is 1, and so on) and then to one-hot vectors.

The **multiple sequence alignment** contains thousands of related sequences, each aligned to the target. The MSA provides the evolutionary context that reveals co-varying positions. Each position in each sequence is encoded with features indicating the amino acid identity plus additional information like insertion counts and deletion states.

Optionally, **template structures** provide hints from experimentally determined structures of related proteins. If the database contains a protein with 40% sequence identity to your target, that structure offers valuable geometric clues.

### Creating Initial Representations

The embedding layer must transform these inputs into the MSA and pair representations that the network will refine:

```python
import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    """Embed input features into MSA and pair representations."""

    def __init__(self, c_m=256, c_z=128):
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z

        # MSA embedding
        self.msa_embedding = nn.Linear(49, c_m)  # 49 = 21 AA + 1 gap + features

        # Pair embedding from sequence
        self.left_single = nn.Linear(21, c_z)
        self.right_single = nn.Linear(21, c_z)

        # Relative position encoding
        self.relpos_embedding = nn.Embedding(65, c_z)  # -32 to +32

    def forward(self, msa_feat, target_feat, residue_index):
        """
        Args:
            msa_feat: [N_seq, L, 49] MSA features
            target_feat: [L, 21] One-hot target sequence
            residue_index: [L] Residue positions
        Returns:
            msa_repr: [N_seq, L, c_m]
            pair_repr: [L, L, c_z]
        """
        # MSA representation
        msa_repr = self.msa_embedding(msa_feat)

        # Pair representation from outer product
        left = self.left_single(target_feat)   # [L, c_z]
        right = self.right_single(target_feat) # [L, c_z]
        pair_repr = left[:, None, :] + right[None, :, :]  # [L, L, c_z]

        # Add relative position encoding
        d = residue_index[:, None] - residue_index[None, :]  # [L, L]
        d = torch.clamp(d + 32, 0, 64)  # Clip to [-32, 32] range
        pair_repr = pair_repr + self.relpos_embedding(d)

        return msa_repr, pair_repr
```

The pair representation starts simple: it combines embeddings of the left and right residues with information about how far apart they are in sequence. Residues that are neighbors in sequence (like positions 15 and 16) get different positional encodings than residues far apart (like positions 15 and 150). This gives the network a starting point, since nearby residues are more likely to be spatially close.

### Key Dimensions

Throughout AlphaFold2, several dimension constants appear repeatedly:

| Representation | Shape | Description |
|----------------|-------|-------------|
| MSA representation | $[N_{seq} \times L \times c_m]$ | Per-sequence, per-residue features |
| Pair representation | $[L \times L \times c_z]$ | Pairwise residue relationships |
| Single representation | $[L \times c_s]$ | Per-residue features for structure |

Where typically: $c_m = 256$, $c_z = 128$, $c_s = 384$

These representations flow through the network, gradually accumulating information until they contain enough knowledge to predict atomic coordinates.

---

## 3. The Evoformer: Where Evolution Meets Attention

The Evoformer is the heart of AlphaFold2, a stack of 48 nearly identical blocks that iteratively refine the MSA and pair representations. The name itself reveals its purpose: it is a transformer designed to process evolutionary information.

What makes the Evoformer special is not its size but its architecture. Every component was designed to capture specific patterns that matter for protein structure. Let us examine each piece.

### 3.1 MSA Row Attention: Learning from Co-evolution

Imagine you are examining a single sequence in the MSA, looking at position 15 and trying to understand its context. Row attention lets each position attend to other positions within the same sequence, learning dependencies along the protein chain.

But here is where AlphaFold2 becomes clever: the attention is biased by the pair representation. When position 15 decides how much attention to pay to position 47, it considers not just their sequence features but also what the pair representation says about their relationship. If the pair representation encodes that these positions are likely in contact, the attention mechanism gives them more opportunity to exchange information.

This creates a beautiful feedback loop. The pair representation influences how the MSA is processed, and later, information from the MSA will update the pair representation. The two representations co-evolve within the network, each informing the other.

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + b_{ij}\right)V$$

The term $b_{ij}$ is the pair bias, a learned projection of the pair representation that nudges attention toward pairs the network believes are structurally related.

```python
class MSARowAttentionWithPairBias(nn.Module):
    """MSA row attention with pair representation as bias."""

    def __init__(self, c_m=256, c_z=128, n_heads=8):
        super().__init__()
        self.c_m = c_m
        self.n_heads = n_heads
        self.head_dim = c_m // n_heads

        self.layer_norm_m = nn.LayerNorm(c_m)
        self.layer_norm_z = nn.LayerNorm(c_z)

        # Q, K, V projections
        self.to_q = nn.Linear(c_m, c_m, bias=False)
        self.to_k = nn.Linear(c_m, c_m, bias=False)
        self.to_v = nn.Linear(c_m, c_m, bias=False)

        # Pair bias projection
        self.pair_bias = nn.Linear(c_z, n_heads, bias=False)

        # Output projection
        self.to_out = nn.Linear(c_m, c_m)
        self.gate = nn.Linear(c_m, c_m)

    def forward(self, msa_repr, pair_repr):
        """
        Args:
            msa_repr: [N_seq, L, c_m]
            pair_repr: [L, L, c_z]
        Returns:
            msa_repr: [N_seq, L, c_m]
        """
        N_seq, L, _ = msa_repr.shape

        # Normalize
        m = self.layer_norm_m(msa_repr)
        z = self.layer_norm_z(pair_repr)

        # Compute Q, K, V
        q = self.to_q(m).view(N_seq, L, self.n_heads, self.head_dim)
        k = self.to_k(m).view(N_seq, L, self.n_heads, self.head_dim)
        v = self.to_v(m).view(N_seq, L, self.n_heads, self.head_dim)

        # Compute attention scores
        # [N_seq, n_heads, L, L]
        attn = torch.einsum('bihd,bjhd->bhij', q, k) / (self.head_dim ** 0.5)

        # Add pair bias: [L, L, n_heads] -> [1, n_heads, L, L]
        bias = self.pair_bias(z).permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias

        # Softmax and apply to values
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        out = out.reshape(N_seq, L, self.c_m)

        # Gated output
        gate = torch.sigmoid(self.gate(m))
        out = gate * self.to_out(out)

        return msa_repr + out
```

Notice the gating mechanism at the end. The network learns when to incorporate new information and when to preserve existing features, a pattern that appears throughout AlphaFold2 and helps with gradient flow during training.

### 3.2 MSA Column Attention: Comparing Across Evolution

While row attention examines relationships along a single sequence, column attention looks at the same position across different sequences in the MSA. If row attention asks, "How does position 15 relate to position 47?", column attention asks, "What do all these different organisms tell us about position 15?"

Column attention is where co-evolutionary signals become explicit. When the network attends across sequences at a given position, it learns patterns like, "whenever this position is hydrophobic, that position tends to be hydrophobic too." These patterns emerge because the network sees thousands of evolutionary experiments, each sequence representing a different solution to the same folding problem.

This is computationally expensive because you must attend over all sequences, which can number in the thousands. AlphaFold2 uses a technique called axial attention, processing rows and columns separately rather than attending over the full MSA at once.

### 3.3 Triangular Updates: Geometry Through Message Passing

Now we come to one of AlphaFold2's most elegant ideas: triangular attention and triangular multiplicative updates. These operations encode a fundamental truth about three-dimensional geometry.

Consider three residues: A, B, and C. If you know that A is close to B (say, within 8 angstroms), and you know that B is close to C, what can you infer about the relationship between A and C? In flat, Euclidean space, you cannot say much, since A and C could be anywhere from 0 to 16 angstroms apart depending on the angle. But proteins are densely packed, and the constraints are much tighter than arbitrary Euclidean geometry suggests.

The triangular operations enforce this kind of three-body consistency. They pass messages around triangles in the pair representation:

**Triangular multiplicative update (outgoing):** For each pair (i, j), aggregate information from all pairs (i, k) and (j, k) that share a common endpoint. If i is near many of the same residues as j, they are probably near each other too.

$$z_{ij} \leftarrow z_{ij} + \sum_k a_{ik} \odot b_{jk}$$

**Triangular multiplicative update (incoming):** The same idea, but aggregating over pairs that point toward i and j rather than away from them.

$$z_{ij} \leftarrow z_{ij} + \sum_k a_{ki} \odot b_{kj}$$

```python
class TriangularMultiplicativeUpdate(nn.Module):
    """Triangular multiplicative update for pair representation."""

    def __init__(self, c_z=128, c_hidden=128, mode='outgoing'):
        super().__init__()
        self.c_z = c_z
        self.mode = mode  # 'outgoing' or 'incoming'

        self.layer_norm = nn.LayerNorm(c_z)

        # Left and right projections
        self.left_proj = nn.Linear(c_z, c_hidden)
        self.right_proj = nn.Linear(c_z, c_hidden)

        # Gates
        self.left_gate = nn.Linear(c_z, c_hidden)
        self.right_gate = nn.Linear(c_z, c_hidden)

        # Output
        self.output_gate = nn.Linear(c_z, c_z)
        self.output_proj = nn.Linear(c_hidden, c_z)
        self.final_norm = nn.LayerNorm(c_hidden)

    def forward(self, pair_repr):
        """
        Args:
            pair_repr: [L, L, c_z]
        Returns:
            pair_repr: [L, L, c_z]
        """
        z = self.layer_norm(pair_repr)

        # Project and gate
        left = self.left_proj(z) * torch.sigmoid(self.left_gate(z))
        right = self.right_proj(z) * torch.sigmoid(self.right_gate(z))

        # Triangular update
        if self.mode == 'outgoing':
            # z_ij += sum_k (a_ik * b_jk)
            # left: [L, L, c], right: [L, L, c]
            # Need: left[i,k,:] * right[j,k,:] summed over k
            out = torch.einsum('ikc,jkc->ijc', left, right)
        else:  # incoming
            # z_ij += sum_k (a_ki * b_kj)
            out = torch.einsum('kic,kjc->ijc', left, right)

        # Output projection with gate
        out = self.final_norm(out)
        out = self.output_proj(out)
        gate = torch.sigmoid(self.output_gate(pair_repr))

        return pair_repr + gate * out
```

The triangular attention operations work similarly but use attention rather than element-wise multiplication to aggregate information. Both starting-node and ending-node variants exist, providing different views of the geometric relationships.

```python
class TriangularAttention(nn.Module):
    """Triangular self-attention for pair representation."""

    def __init__(self, c_z=128, n_heads=4, mode='starting'):
        super().__init__()
        self.c_z = c_z
        self.n_heads = n_heads
        self.head_dim = c_z // n_heads
        self.mode = mode  # 'starting' or 'ending'

        self.layer_norm = nn.LayerNorm(c_z)

        self.to_q = nn.Linear(c_z, c_z, bias=False)
        self.to_k = nn.Linear(c_z, c_z, bias=False)
        self.to_v = nn.Linear(c_z, c_z, bias=False)

        # Attention bias from pair representation
        self.bias_proj = nn.Linear(c_z, n_heads, bias=False)

        self.to_out = nn.Linear(c_z, c_z)
        self.gate = nn.Linear(c_z, c_z)

    def forward(self, pair_repr):
        """
        Args:
            pair_repr: [L, L, c_z]
        Returns:
            pair_repr: [L, L, c_z]
        """
        L = pair_repr.shape[0]

        if self.mode == 'ending':
            # Transpose to operate on columns
            pair_repr = pair_repr.transpose(0, 1)

        z = self.layer_norm(pair_repr)

        # [L, L, c_z] -> [L, L, n_heads, head_dim]
        q = self.to_q(z).view(L, L, self.n_heads, self.head_dim)
        k = self.to_k(z).view(L, L, self.n_heads, self.head_dim)
        v = self.to_v(z).view(L, L, self.n_heads, self.head_dim)

        # Attention: for each row i, attend over columns j
        # q[i,j], k[i,k] -> attn[i,j,k]
        attn = torch.einsum('ijhd,ikhd->hijk', q, k) / (self.head_dim ** 0.5)

        # Bias from pair representation
        # For starting: bias from z[j,k]
        bias = self.bias_proj(z)  # [L, L, n_heads]
        attn = attn + bias.permute(2, 0, 1).unsqueeze(1)  # broadcast over j

        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('hijk,ikhd->ijhd', attn, v)
        out = out.reshape(L, L, self.c_z)

        gate = torch.sigmoid(self.gate(pair_repr))
        out = gate * self.to_out(out)
        result = pair_repr + out

        if self.mode == 'ending':
            result = result.transpose(0, 1)

        return result
```

Why both multiplicative updates and attention? They capture different aspects of geometric constraints. The multiplicative updates are like hard constraints, directly computing products that aggregate over triangles. The attention operations are softer, letting the network learn which triangles matter most for each pair.

### 3.4 Outer Product Mean: Bridging MSA and Pairs

The MSA representation and pair representation need to communicate. The outer product mean is the primary pathway from MSA to pairs.

The intuition is straightforward: if two positions have similar patterns across the MSA, they are probably related. The outer product literally computes this correlation:

$$z_{ij} \leftarrow z_{ij} + \frac{1}{N_{seq}} \sum_s m_{si} \otimes m_{sj}$$

For each pair (i, j), we take the MSA features at position i and position j, compute their outer product, and average over all sequences. This creates a dense representation of how these positions co-vary across evolution.

```python
class OuterProductMean(nn.Module):
    """Project MSA representation to pair representation via outer product."""

    def __init__(self, c_m=256, c_z=128, c_hidden=32):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_m)
        self.left_proj = nn.Linear(c_m, c_hidden)
        self.right_proj = nn.Linear(c_m, c_hidden)
        self.output = nn.Linear(c_hidden * c_hidden, c_z)

    def forward(self, msa_repr):
        """
        Args:
            msa_repr: [N_seq, L, c_m]
        Returns:
            pair_update: [L, L, c_z]
        """
        m = self.layer_norm(msa_repr)

        # Project
        left = self.left_proj(m)   # [N_seq, L, c_hidden]
        right = self.right_proj(m) # [N_seq, L, c_hidden]

        # Outer product and mean over sequences
        # [N_seq, L, c] x [N_seq, L, c] -> [L, L, c*c]
        outer = torch.einsum('sic,sjd->ijcd', left, right)
        outer = outer / msa_repr.shape[0]  # Mean over sequences

        # Flatten and project
        outer = outer.reshape(outer.shape[0], outer.shape[1], -1)
        return self.output(outer)
```

### The Complete Evoformer Block

Each Evoformer block orchestrates all these components. The MSA representation passes through row attention, column attention, and transition layers. The pair representation passes through triangular multiplicative updates, triangular attention (both starting and ending node variants), and its own transition layers. The outer product mean bridges them at the end.

```
+-------------------------------------------------------------+
|                      Evoformer Block                         |
+-------------------------------------------------------------+
|                                                              |
|  MSA Stack:                    Pair Stack:                   |
|  +--------------------+       +--------------------+        |
|  | MSA Row Attention  |       | Triangular Mult.   |        |
|  | (with pair bias)   |       | (outgoing)         |        |
|  +--------------------+       +--------------------+        |
|           |                            |                     |
|           v                            v                     |
|  +--------------------+       +--------------------+        |
|  | MSA Column Attention       | Triangular Mult.   |        |
|  | (global)           |       | (incoming)         |        |
|  +--------------------+       +--------------------+        |
|           |                            |                     |
|           v                            v                     |
|  +--------------------+       +--------------------+        |
|  | MSA Transition     |       | Triangular Attn.   |        |
|  | (feed-forward)     |       | (starting node)    |        |
|  +--------------------+       +--------------------+        |
|           |                            |                     |
|           |                            v                     |
|           |               +--------------------+            |
|           |               | Triangular Attn.   |            |
|           |               | (ending node)      |            |
|           |               +--------------------+            |
|           |                            |                     |
|           |                            v                     |
|           |               +--------------------+            |
|           |               | Pair Transition    |            |
|           |               +--------------------+            |
|           |                            |                     |
|           v                            v                     |
|  +--------------------------------------------+             |
|  |         Outer Product Mean (MSA -> Pair)    |             |
|  +--------------------------------------------+             |
|                                                              |
+-------------------------------------------------------------+
```

After 48 of these blocks, the representations have been refined through thousands of attention operations. The pair representation now contains detailed information about spatial relationships, essentially a predicted distance map. But we still need to convert this into actual 3D coordinates.

---

## 4. The Structure Module: From Features to Coordinates

The Structure Module is where AlphaFold2 produces its final output: atomic coordinates for every residue. This is arguably the most innovative part of the architecture, introducing Invariant Point Attention (IPA), a mechanism that has since influenced many other geometric deep learning models.

### 4.1 The Challenge of 3D Structure

Here is the fundamental problem: protein structures exist in three-dimensional space, and that space has symmetries. If you rotate a protein by 90 degrees, it is still the same protein. If you translate it 10 angstroms to the left, the structure is unchanged. Any valid structure prediction method must respect these symmetries, a property called SE(3) invariance (or equivariance, depending on context).

Earlier deep learning approaches for proteins often predicted distance matrices or contact maps rather than coordinates precisely because distances are naturally invariant to rotation and translation. But AlphaFold2 wanted to predict actual coordinates, which meant building invariance into the architecture itself.

### 4.2 Frames: A Language for Protein Geometry

AlphaFold2's solution is elegant: represent each residue as a rigid body frame, a local coordinate system defined by a rotation and a translation. The backbone atoms of each residue (N, C-alpha, C) define a natural reference frame. The translation locates where the residue is in space; the rotation describes how it is oriented.

A frame $T_i = (R_i, \vec{t}_i)$ consists of:
- $R_i \in SO(3)$: a rotation matrix (3 by 3, orthogonal, determinant 1)
- $\vec{t}_i \in \mathbb{R}^3$: a translation vector

Any point in space can be expressed either in global coordinates or in the local coordinate system of any frame. Converting between them is fundamental to how the Structure Module operates.

```python
import torch
from scipy.spatial.transform import Rotation

class Rigid:
    """Rigid body transformation (rotation + translation)."""

    def __init__(self, rots, trans):
        """
        Args:
            rots: [*, 3, 3] rotation matrices
            trans: [*, 3] translation vectors
        """
        self.rots = rots
        self.trans = trans

    @staticmethod
    def identity(shape, device='cpu'):
        """Create identity transformation."""
        rots = torch.eye(3, device=device).expand(*shape, 3, 3).clone()
        trans = torch.zeros(*shape, 3, device=device)
        return Rigid(rots, trans)

    def compose(self, other):
        """Compose two rigid transformations: self * other."""
        new_rots = torch.einsum('...ij,...jk->...ik', self.rots, other.rots)
        new_trans = torch.einsum('...ij,...j->...i', self.rots, other.trans) + self.trans
        return Rigid(new_rots, new_trans)

    def apply(self, points):
        """Apply transformation to points: R @ x + t."""
        return torch.einsum('...ij,...j->...i', self.rots, points) + self.trans

    def invert(self):
        """Compute inverse transformation."""
        inv_rots = self.rots.transpose(-1, -2)
        inv_trans = -torch.einsum('...ij,...j->...i', inv_rots, self.trans)
        return Rigid(inv_rots, inv_trans)

    def to_tensor_4x4(self):
        """Convert to 4x4 homogeneous matrix."""
        shape = self.rots.shape[:-2]
        mat = torch.zeros(*shape, 4, 4, device=self.rots.device)
        mat[..., :3, :3] = self.rots
        mat[..., :3, 3] = self.trans
        mat[..., 3, 3] = 1
        return mat
```

### 4.3 Invariant Point Attention: The Crown Jewel

Invariant Point Attention is perhaps the most important architectural innovation in AlphaFold2. It is the mechanism that allows the network to reason about three-dimensional geometry while maintaining invariance to global rotations and translations.

The key insight is that attention can incorporate 3D information in an invariant way by working in local coordinate frames. Here is how it works:

Standard attention computes similarity between queries and keys, both of which are learned projections of input features. IPA extends this by including point queries and point keys, 3D coordinates expressed in each residue's local frame.

When residue i attends to residue j, the attention score includes three components:

1. **Scalar attention**: Standard query-key similarity on feature vectors.
2. **Pair bias**: Information from the pair representation about the relationship between i and j.
3. **Point attention**: The distance between points when expressed in a common reference frame.

The magic is in component 3. Each residue generates query points and key points in its local coordinate system. To compare them, we transform residue j's key points into residue i's coordinate frame. The squared distance between query points and transformed key points contributes to the attention score. Crucially, this distance is invariant to global rotations and translations because both sets of points are expressed relative to the same local frame.

$$a_{ij} = \underbrace{q_i \cdot k_j}_{\text{sequence}} + \underbrace{b_{ij}}_{\text{pair bias}} + \underbrace{\sum_p \|T_i^{-1}(T_j(\vec{p}_{jp})) - \vec{q}_{ip}\|^2}_{\text{spatial distance}}$$

The attended values similarly include scalar features, pair features, and point features. The output point features are transformed back to each residue's local frame, ensuring the entire operation respects SE(3) symmetry.

```python
class InvariantPointAttention(nn.Module):
    """Invariant Point Attention - SE(3) equivariant attention."""

    def __init__(self, c_s=384, c_z=128, n_heads=12, n_qk_points=4, n_v_points=8):
        super().__init__()
        self.c_s = c_s
        self.n_heads = n_heads
        self.head_dim = c_s // n_heads
        self.n_qk_points = n_qk_points
        self.n_v_points = n_v_points

        # Scalar attention (standard)
        self.to_q = nn.Linear(c_s, c_s, bias=False)
        self.to_k = nn.Linear(c_s, c_s, bias=False)
        self.to_v = nn.Linear(c_s, c_s, bias=False)

        # Point attention (3D coordinates in local frame)
        self.to_q_points = nn.Linear(c_s, n_heads * n_qk_points * 3)
        self.to_k_points = nn.Linear(c_s, n_heads * n_qk_points * 3)
        self.to_v_points = nn.Linear(c_s, n_heads * n_v_points * 3)

        # Pair bias
        self.pair_bias = nn.Linear(c_z, n_heads, bias=False)

        # Learnable weights for point attention
        self.head_weights = nn.Parameter(torch.zeros(n_heads))

        # Output projection
        out_dim = c_s + n_heads * n_v_points * 3 + n_heads * c_z
        self.to_out = nn.Linear(out_dim, c_s)

    def forward(self, single_repr, pair_repr, rigids):
        """
        Args:
            single_repr: [L, c_s] per-residue features
            pair_repr: [L, L, c_z] pairwise features
            rigids: Rigid object with [L] frames
        Returns:
            single_repr: [L, c_s] updated features
        """
        L = single_repr.shape[0]

        # Scalar queries, keys, values
        q = self.to_q(single_repr).view(L, self.n_heads, self.head_dim)
        k = self.to_k(single_repr).view(L, self.n_heads, self.head_dim)
        v = self.to_v(single_repr).view(L, self.n_heads, self.head_dim)

        # Point queries, keys, values (in local frames)
        q_pts = self.to_q_points(single_repr).view(L, self.n_heads, self.n_qk_points, 3)
        k_pts = self.to_k_points(single_repr).view(L, self.n_heads, self.n_qk_points, 3)
        v_pts = self.to_v_points(single_repr).view(L, self.n_heads, self.n_v_points, 3)

        # Transform points to global frame
        q_pts_global = rigids.apply(q_pts.reshape(L, -1, 3)).view(L, self.n_heads, self.n_qk_points, 3)
        k_pts_global = rigids.apply(k_pts.reshape(L, -1, 3)).view(L, self.n_heads, self.n_qk_points, 3)
        v_pts_global = rigids.apply(v_pts.reshape(L, -1, 3)).view(L, self.n_heads, self.n_v_points, 3)

        # Compute attention logits
        # 1. Scalar attention
        attn_scalar = torch.einsum('ihd,jhd->hij', q, k) / (self.head_dim ** 0.5)

        # 2. Point attention (squared distance)
        pt_diff = q_pts_global[:, None, :, :, :] - k_pts_global[None, :, :, :, :]  # [L, L, H, P, 3]
        pt_dist_sq = (pt_diff ** 2).sum(dim=-1).sum(dim=-1)  # [L, L, H]

        # Weight by learnable head weights
        w_c = torch.softplus(self.head_weights)
        attn_points = -0.5 * w_c * pt_dist_sq.permute(2, 0, 1)  # [H, L, L]

        # 3. Pair bias
        attn_pair = self.pair_bias(pair_repr).permute(2, 0, 1)  # [H, L, L]

        # Combine attention logits
        attn = attn_scalar.permute(2, 0, 1) + attn_points + attn_pair
        attn = torch.softmax(attn, dim=-1)  # [H, L, L]

        # Apply attention to values
        # Scalar output
        out_scalar = torch.einsum('hij,jhd->ihd', attn, v)
        out_scalar = out_scalar.reshape(L, self.c_s)

        # Point output (in global frame, then transform to local)
        out_pts_global = torch.einsum('hij,jhpc->ihpc', attn, v_pts_global)
        out_pts_local = rigids.invert().apply(out_pts_global.reshape(L, -1, 3))
        out_pts = out_pts_local.reshape(L, -1)

        # Pair output
        out_pair = torch.einsum('hij,ijc->ihc', attn, pair_repr).reshape(L, -1)

        # Concatenate and project
        out = torch.cat([out_scalar, out_pts, out_pair], dim=-1)
        return self.to_out(out)
```

### 4.4 Iterative Refinement

The Structure Module does not predict coordinates in a single pass. Instead, it initializes all residue frames to identity (placing every residue at the origin with default orientation) and iteratively refines them through multiple layers of IPA.

Each iteration:
1. Applies IPA to update per-residue features
2. Passes features through a transition network
3. Predicts updates to each residue's frame (small rotations and translations)
4. Composes these updates with the current frames

This iterative approach is reminiscent of message passing in graph neural networks, where information propagates through local interactions over multiple rounds. Early iterations establish coarse global structure; later iterations fine-tune local geometry.

```python
class StructureModule(nn.Module):
    """Structure Module: converts representations to 3D structure."""

    def __init__(self, c_s=384, c_z=128, n_layers=8):
        super().__init__()

        # Project MSA to single representation
        self.input_proj = nn.Linear(256, c_s)  # c_m -> c_s

        # IPA layers
        self.ipa_layers = nn.ModuleList([
            InvariantPointAttention(c_s, c_z) for _ in range(n_layers)
        ])

        # Backbone update (predict frame updates)
        self.backbone_update = nn.Linear(c_s, 6)  # 3 rotation + 3 translation

        # Transition layers
        self.transitions = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(c_s),
                nn.Linear(c_s, c_s * 4),
                nn.ReLU(),
                nn.Linear(c_s * 4, c_s)
            ) for _ in range(n_layers)
        ])

    def forward(self, msa_repr, pair_repr):
        """
        Args:
            msa_repr: [N_seq, L, c_m]
            pair_repr: [L, L, c_z]
        Returns:
            coords: [L, 3] C-alpha coordinates
            frames: Rigid frames for each residue
        """
        L = pair_repr.shape[0]

        # Initialize single representation from first MSA row
        single = self.input_proj(msa_repr[0])  # [L, c_s]

        # Initialize frames as identity
        frames = Rigid.identity((L,), device=single.device)

        # Iterative refinement
        for ipa, transition in zip(self.ipa_layers, self.transitions):
            # IPA attention
            single = single + ipa(single, pair_repr, frames)

            # Transition
            single = single + transition(single)

            # Update frames
            update = self.backbone_update(single)

            # Convert to rotation (simplified - use quaternion in practice)
            rot_update = update[:, :3]  # Small rotation angles
            trans_update = update[:, 3:]

            # Create rotation matrix from angles (simplified)
            angles = rot_update * 0.1  # Scale down
            cos_a, sin_a = torch.cos(angles), torch.sin(angles)

            # Simplified rotation update (proper implementation uses quaternions)
            rot_mat = torch.eye(3, device=single.device).unsqueeze(0).expand(L, -1, -1).clone()

            frame_update = Rigid(rot_mat, trans_update)
            frames = frames.compose(frame_update)

        # Extract C-alpha coordinates (origin of each frame)
        coords = frames.trans

        return coords, frames
```

---

## 5. FAPE Loss: Teaching Geometry Through Local Frames

We have seen how AlphaFold2 represents and predicts structure. Now we need to understand how it learns, which requires a loss function that captures what it means for a predicted structure to be correct.

### The Problem with RMSD

The most intuitive metric for comparing two protein structures is root-mean-square deviation (RMSD): superimpose the structures optimally and measure the average squared displacement of corresponding atoms. RMSD has been the standard metric in structural biology for decades.

But RMSD has problems as a training loss. First, it requires optimal superposition, which involves finding the rotation that minimizes the error. This is not differentiable in a clean way that works well with gradient descent. Second, RMSD treats all errors equally. An error of 2 angstroms in a floppy loop is just as bad as a 2 angstrom error in a rigid beta sheet, even though the loop error might be physically reasonable while the sheet error indicates a fundamental mistake.

### Frame Aligned Point Error

AlphaFold2's solution is Frame Aligned Point Error, or FAPE. The insight is beautiful: instead of measuring error in global coordinates, measure it in local coordinate frames.

For each residue i with its local frame, we express the positions of all other residues j in that frame. We do this for both the predicted structure and the true structure. The FAPE loss is the average error between these local coordinates:

$$\text{FAPE} = \frac{1}{L^2} \sum_{i,j} \left\| T_i^{-1}(\vec{x}_j) - T_i^{pred,-1}(\vec{x}_j^{pred}) \right\|$$

Why is this genius? First, it is automatically SE(3) invariant. Global rotations and translations cancel out because we are always measuring relative to local frames. Second, it emphasizes local accuracy. If the loop is somewhat mobile but locally correct, errors in one part of the loop do not propagate to affect the loss at distant residues. Third, it provides rich gradient signal because every pair of residues contributes independently to the loss.

```python
def fape_loss(pred_frames, pred_coords, true_frames, true_coords, clamp_distance=10.0):
    """
    Frame Aligned Point Error loss.

    Args:
        pred_frames: Rigid - predicted frames [L]
        pred_coords: [L, 3] - predicted coordinates
        true_frames: Rigid - ground truth frames [L]
        true_coords: [L, 3] - ground truth coordinates
        clamp_distance: Maximum distance to clamp (for robustness)

    Returns:
        loss: Scalar FAPE loss
    """
    L = pred_coords.shape[0]

    # Compute local coordinates in each frame
    # For frame i, transform all coordinates j

    # Predicted local coords: T_i^{-1} @ x_j
    pred_inv = pred_frames.invert()
    # [L, L, 3] - for each frame i, coordinates of all atoms j
    pred_local = pred_inv.apply(pred_coords.unsqueeze(0).expand(L, -1, -1).reshape(-1, 3))
    pred_local = pred_local.view(L, L, 3)

    # True local coords
    true_inv = true_frames.invert()
    true_local = true_inv.apply(true_coords.unsqueeze(0).expand(L, -1, -1).reshape(-1, 3))
    true_local = true_local.view(L, L, 3)

    # Compute error
    error = torch.sqrt(((pred_local - true_local) ** 2).sum(dim=-1) + 1e-8)

    # Clamp for robustness
    error = torch.clamp(error, max=clamp_distance)

    return error.mean()
```

The clamping at 10 angstroms prevents outliers from dominating the gradient. If part of the structure is completely wrong, the loss stops increasing beyond the clamp, allowing the network to focus on fixing correctible errors.

### Multi-Task Learning

AlphaFold2 does not rely on FAPE alone. It uses multiple auxiliary losses that provide additional training signal:

**Distogram loss** predicts pairwise distance distributions. For each pair of residues, the network predicts probabilities over discretized distance bins. This provides dense supervision over all pairs, not just at the final coordinate level.

**pLDDT loss** trains a confidence head to predict per-residue accuracy. The network learns to know when it does not know, which is valuable for downstream users assessing prediction reliability.

**Masked MSA loss** reconstructs randomly masked positions in the MSA, similar to BERT's masked language modeling. This encourages the network to learn meaningful representations of evolutionary information.

```python
def alphafold_loss(predictions, targets, config):
    """
    Combined AlphaFold2 loss function.

    Args:
        predictions: dict with 'frames', 'coords', 'distogram', 'plddt'
        targets: dict with ground truth values
        config: loss weights
    """
    losses = {}

    # FAPE loss (main structural loss)
    losses['fape'] = fape_loss(
        predictions['frames'],
        predictions['coords'],
        targets['frames'],
        targets['coords']
    )

    # Distogram loss (auxiliary)
    if 'distogram' in predictions:
        # Cross-entropy over distance bins
        distogram_pred = predictions['distogram']  # [L, L, n_bins]
        distogram_true = targets['distogram']       # [L, L] bin indices
        losses['distogram'] = nn.functional.cross_entropy(
            distogram_pred.reshape(-1, distogram_pred.shape[-1]),
            distogram_true.reshape(-1)
        )

    # pLDDT loss (confidence prediction)
    if 'plddt' in predictions:
        plddt_pred = predictions['plddt']  # [L]
        plddt_true = compute_lddt(predictions['coords'], targets['coords'])
        losses['plddt'] = nn.functional.mse_loss(plddt_pred, plddt_true)

    # Combine with weights
    total_loss = sum(config.get(k, 1.0) * v for k, v in losses.items())

    return total_loss, losses
```

---

## 6. Putting It All Together

Let us step back and see how all the pieces fit together in the complete AlphaFold2 pipeline.

```
+------------------------------------------------------------------+
|                        AlphaFold2 Pipeline                        |
+------------------------------------------------------------------+
|                                                                   |
|  +------------+    +-------------+    +---------------+    +----+ |
|  | Input      |--->| Embedding   |--->|  Evoformer    |--->|Struct| |
|  | Features   |    |  Layer      |    |  (48 blocks)  |    |Module| |
|  +------------+    +-------------+    +---------------+    +----+ |
|       |                                                      |    |
|       |         MSA: [N_seq x L x c_m]                       |    |
|       |         Pair: [L x L x c_z]                          v    |
|       |                                              +----------+ |
|       +--------------------------------------------->| 3D Coords| |
|                                                      +----------+ |
+------------------------------------------------------------------+
```

**Step 1: Input preparation.** Search databases for related sequences to build an MSA. Optionally find template structures from related proteins with known structures.

**Step 2: Embedding.** Convert MSA features and sequence identity into initial representations. The MSA representation captures per-position, per-sequence information. The pair representation starts with simple sequence-based features plus relative position encodings.

**Step 3: Evoformer processing.** Run 48 Evoformer blocks. Each block updates MSA features through row and column attention, updates pair features through triangular operations, and uses outer product mean to transfer evolutionary information from MSA to pairs. By the end, the pair representation encodes detailed distance relationships.

**Step 4: Structure prediction.** The Structure Module takes the refined representations and iteratively builds 3D structure through Invariant Point Attention. Each of 8 iterations updates residue frames, gradually moving from identity (everything at the origin) to the final predicted structure.

**Step 5: Output and confidence.** Return predicted coordinates along with pLDDT confidence scores for each residue and PAE (predicted aligned error) for each residue pair.

---

## 7. Design Principles and Lessons

AlphaFold2's success was not accidental. Every component reflects principled thinking about what information matters for protein structure and how neural networks can extract it.

### Use Evolutionary Information

Proteins are not designed from scratch; they evolve from ancestors. Related sequences are experiments run by evolution, each revealing something about the constraints on structure. AlphaFold2 makes MSA processing central, using specialized attention mechanisms to extract co-evolutionary signals.

### Model Pairwise Relationships Explicitly

Protein structure is fundamentally about which residues are near which other residues. The pair representation makes these relationships explicit, and triangular operations enforce geometric consistency.

### Respect Symmetry

Three-dimensional space has rotational and translational symmetry. Rather than hoping the network learns to ignore arbitrary coordinate choices, AlphaFold2 builds invariance into the architecture through local frames and IPA.

### Iterate and Refine

Complex problems benefit from iterative refinement. The Evoformer runs 48 blocks; the Structure Module runs 8 iterations. Each pass improves on the last, allowing local decisions to propagate globally.

### Learn What You Do Not Know

The pLDDT confidence head is not an afterthought. AlphaFold2 explicitly models its own uncertainty, distinguishing confident predictions from guesses. This has proven enormously valuable for users interpreting results.

| Principle | Implementation |
|-----------|----------------|
| Use evolutionary information | MSA processing in Evoformer |
| Model pairwise relationships | Pair representation + triangular updates |
| SE(3) equivariance | IPA, frame-based representation |
| Iterative refinement | 8 iterations in Structure Module |
| Multi-task learning | Multiple loss terms (FAPE, distogram, pLDDT) |

---

## 8. Computational Considerations

AlphaFold2 is computationally demanding. Understanding the bottlenecks helps when implementing or adapting the architecture.

**Memory scales quadratically with sequence length** because of the pair representation. A 1000-residue protein requires storing a 1000 by 1000 by 128 tensor for pair features, plus similar arrays for attention computations.

**MSA column attention** is expensive because it attends over potentially thousands of sequences. AlphaFold2 samples down to 512 sequences for the extra MSA stack.

**Triangular attention** has cubic complexity in sequence length because it computes attention over rows/columns of the pair representation with biases from the full matrix.

In practice:
- Large proteins require chunked processing, splitting the sequence into overlapping windows
- Mixed precision training (FP16 or BF16) reduces memory significantly
- Gradient checkpointing trades compute for memory by recomputing activations during the backward pass

---

## 9. Practice Problems

1. **Implement MSA column attention**: Extend the row attention code to operate over sequences instead of positions. What changes in the einsum operations?

2. **Quaternion rotations**: The Structure Module code uses a simplified rotation update. Implement proper quaternion-based rotations that correctly compose small angular updates.

3. **Template embedding**: AlphaFold2 incorporates structural templates. Design an embedding scheme that converts template backbone coordinates into pair representation features.

4. **Confidence prediction**: Implement a pLDDT prediction head that takes the single representation from the Structure Module and predicts per-residue confidence.

---

## 10. Key Takeaways

AlphaFold2 solved protein structure prediction by combining several ideas:

**Evolutionary information is paramount.** The MSA is not just an input feature; it is the primary source of structural information. AlphaFold2's architecture is designed around extracting and processing evolutionary signals.

**Geometry has grammar.** Triangular operations enforce that pairwise relationships are geometrically consistent. You cannot have A near B, B near C, and A far from C indefinitely; the network learns this constraint.

**Symmetry should be built in, not learned.** Invariant Point Attention is SE(3) equivariant by construction, not by hoping the network discovers rotational symmetry from data.

**Good loss functions embed domain knowledge.** FAPE measures error in local coordinate frames, naturally handling the fact that some regions are more rigid than others.

The full AlphaFold2 system involves additional components we have not covered in depth: recycling (running the network multiple times with outputs fed back as inputs), MSA row clustering, template processing, side-chain prediction, and more. But the core ideas, Evoformer, IPA, frames, and FAPE, capture what makes the architecture tick.

Understanding AlphaFold2 is not just historical interest. These ideas underpin many subsequent developments: AlphaFold-Multimer for complexes, RoseTTAFold and ESMFold's variations on the theme, and diffusion models for protein design that build on frame-based representations. The principles here will serve you well as you explore the rapidly evolving landscape of AI for proteins.

---

## 11. Further Reading

1. Jumper et al. (2021). "Highly accurate protein structure prediction with AlphaFold." *Nature*. The original paper with full architecture details.

2. AlphaFold2 Supplementary Information. Detailed pseudocode and diagrams for every component. Essential reading for implementers.

3. OpenFold. Open-source reimplementation with extensive documentation and training code. Available at github.com/aqlaboratory/openfold.

4. ColabFold. Accessible AlphaFold2 predictions through Google Colab, including faster MSA generation. Available at github.com/sokrypton/ColabFold.

5. Lin et al. (2023). "Evolutionary-scale prediction of atomic-level protein structure with a language model." Describes ESMFold, which achieves similar accuracy without MSAs by using large protein language models.

---

## Summary

AlphaFold2's success comes from three interlocking innovations:

**The Evoformer** processes co-evolutionary information through specialized attention mechanisms, extracting structural signals from thousands of related sequences and enforcing geometric consistency through triangular operations.

**The Structure Module** uses SE(3)-invariant operations to predict 3D coordinates, working with local coordinate frames rather than arbitrary global coordinates, and refining predictions through multiple iterations of Invariant Point Attention.

**The FAPE loss** provides a geometrically meaningful training signal that is invariant to global rotations and translations while emphasizing local accuracy.

Together, these components demonstrate how domain knowledge, about protein evolution, geometry, and symmetry, can be encoded into neural network architecture. AlphaFold2 did not succeed by brute force; it succeeded by understanding proteins.
