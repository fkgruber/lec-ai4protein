# Lecture 5: Neural Network Architectures II - Transformers and Graph Neural Networks

## Learning Objectives

By the end of this lecture, you will be able to:
1. Understand the attention mechanism and its variants
2. Build transformer architectures for protein sequences
3. Implement graph neural networks for protein structures
4. Apply GCN, GAT, and MPNN architectures to proteins

## Prerequisites

- Lecture 4: Neural Network Architectures I - CNNs and RNNs
- Understanding of matrix operations and softmax
- Basic graph theory concepts (nodes, edges, adjacency)

---

## How Do Proteins Teach Us About Each Other?

Imagine you are a protein. You have existed in various forms for billions of years, surviving the relentless pressure of natural selection. You have cousins scattered across every domain of life, from bacteria to whales to oak trees. Each of these relatives carries a slightly different version of your sequence, shaped by the unique survival challenges their hosts faced. Now here is the remarkable thing: by studying this vast family reunion of protein sequences, we can learn which parts of you matter most and which positions are free to change.

This is the profound insight behind co-evolution, and it turns out to be the conceptual foundation for one of the most important innovations in modern deep learning: the attention mechanism.

When biologists compare related protein sequences across species, they notice something fascinating. Certain pairs of positions tend to change together. If position 23 mutates from alanine to valine in one lineage, position 87 often changes from glutamate to aspartate in the same lineage. These correlated mutations are not coincidences. They reveal that positions 23 and 87 are physically close in the folded protein structure, or that they participate in the same functional interaction. When one changes, the other must compensate to maintain the protein's function.

This co-evolutionary signal is so powerful that for decades, computational biologists built entire methods around it, from correlated mutation analysis to direct coupling analysis. These methods asked a simple question: which positions in a protein sequence are paying attention to each other?

The attention mechanism in neural networks formalizes this same intuition. Instead of looking for statistical correlations in evolutionary data, we let the network learn which positions should attend to each other directly from the data. And just as co-evolution reveals the hidden 3D structure lurking within 1D sequences, attention allows neural networks to discover long-range relationships that sequential processing would miss.

---

## The Limitations of Sequential Processing

Before we dive into attention, let us understand why we need it in the first place. In our previous lecture, we explored recurrent neural networks, which process sequences one element at a time, passing information from each step to the next through a hidden state. This approach has an elegant biological intuition: reading a protein sequence is like walking along the chain from the N-terminus to the C-terminus, keeping track of what you have seen so far.

But this sequential processing creates three serious problems.

First, there is the parallelization bottleneck. To compute the hidden state at position 100, you must first compute the hidden states at positions 1 through 99. This fundamentally sequential nature means you cannot take full advantage of modern parallel hardware like GPUs, which shine when they can perform many independent operations simultaneously.

Second, there is the long-range dependency problem. Information from the beginning of a sequence must pass through dozens or hundreds of intermediate steps before it can influence the processing of positions near the end. Each step is an opportunity for information to be lost or distorted. Even with sophisticated gating mechanisms like LSTMs and GRUs, RNNs struggle to capture relationships between distant positions.

Third, there is the information bottleneck. The entire context of a long sequence must be compressed into a fixed-size hidden state vector. For a 500-residue protein, all the information about the first 400 residues must somehow fit into perhaps 256 or 512 numbers before it can influence the processing of residue 401. Important details inevitably get lost in this compression.

Attention offers a radical alternative: what if we let every position communicate directly with every other position? Instead of passing information through a long chain of intermediate steps, we create direct connections. Position 1 can talk to position 500. Position 237 can talk to position 238. Every pairwise relationship is available, and the network learns which relationships matter.

---

## Building Attention from the Ground Up

Imagine you are reading a protein sequence, and you encounter an amino acid at position 50. You want to understand this position in context. What other parts of the sequence should you pay attention to?

Perhaps position 50 is a cysteine, and there is another cysteine at position 127. If these two cysteines form a disulfide bond, they are critically important to each other despite being 77 positions apart in the sequence. You should pay strong attention to position 127.

Or perhaps position 50 is part of a catalytic triad in an enzyme, and the other two members of the triad are at positions 95 and 143. Again, these distant positions are functionally coupled, and you should attend to them.

The attention mechanism captures this intuition mathematically. For each position, we compute three things: a query, a key, and a value. These strange names come from information retrieval, but they have beautiful interpretations in the protein context.

The **query** represents what a position is looking for. Think of it as asking, "What kind of interaction partners am I seeking?" A cysteine's query might implicitly encode, "I'm looking for another cysteine that could form a disulfide bond with me."

The **key** represents what a position has to offer. Think of it as advertising, "Here is what I am and what I can provide." That other cysteine at position 127 has a key that advertises, "I'm a cysteine, potentially available for bonding."

The **value** represents the actual information that gets transmitted when attention is paid. Once position 50 decides to attend to position 127, what information should flow? The value carries this content.

Here is how this works mathematically. We start with an input representation for each position, typically a vector of numbers encoding the amino acid identity and any other features. Let us call the input at position $i$ simply $x_i$. We then compute three vectors for each position:

- Query: $q_i = W^Q x_i$
- Key: $k_i = W^K x_i$
- Value: $v_i = W^V x_i$

The matrices $W^Q$, $W^K$, and $W^V$ are learnable parameters that transform the input into query, key, and value spaces. These transformations are crucial because they allow the network to learn what should look for what.

Next, we compute how much position $i$ should attend to position $j$. We do this by taking the dot product of the query from $i$ and the key from $j$:

$$\text{score}_{ij} = q_i \cdot k_j = x_i^T (W^Q)^T W^K x_j$$

This dot product measures similarity: if the query and key point in similar directions in the transformed space, the score is high, indicating strong attention. If they point in different directions, the score is low.

We then normalize these scores using the softmax function, which converts them into a probability distribution:

$$\alpha_{ij} = \frac{\exp(\text{score}_{ij})}{\sum_k \exp(\text{score}_{ik})}$$

These attention weights $\alpha_{ij}$ sum to 1 across all positions $j$ that position $i$ attends to. They represent a soft selection: position 50 might attend 40% to position 127, 30% to position 95, 20% to position 143, and divide the remaining 10% among other positions.

Finally, we compute the output for position $i$ as a weighted sum of the values:

$$\text{output}_i = \sum_j \alpha_{ij} v_j$$

The positions with high attention weights contribute more to the output. Position 50's new representation is now informed by the representations of the positions it attended to, weighted by how much attention it paid to each.

---

## Scaled Dot-Product Attention

There is one detail we glossed over in our intuitive explanation: the scaling factor. When we compute attention scores using dot products, these scores can become very large if the vectors have many dimensions. Large scores lead to very peaked softmax distributions, which cause gradients to vanish during training.

The solution is to scale the scores by the square root of the key dimension:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This scaling ensures that regardless of the dimension $d_k$, the variance of the dot products remains roughly constant, leading to stable training.

Let us implement this in Python:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        query: (batch, n_heads, seq_len, d_k)
        key: (batch, n_heads, seq_len, d_k)
        value: (batch, n_heads, seq_len, d_v)
        mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)

    Returns:
        output: (batch, n_heads, seq_len, d_v)
        attention_weights: (batch, n_heads, seq_len, seq_len)
    """
    d_k = query.size(-1)

    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: (batch, n_heads, seq_len, seq_len)

    # Apply mask (optional)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights
```

---

## Multi-Head Attention: Different Heads for Different Relationships

When we think about a protein sequence, there are many different types of relationships that might matter. Position 50 might need to attend to:

- Nearby positions for local secondary structure context
- Distant cysteines for potential disulfide bonds
- Positions with complementary chemical properties for the hydrophobic core
- Positions that co-evolve together across species

A single attention mechanism with one set of query, key, and value transformations might struggle to capture all these different relationship types simultaneously. The solution is multi-head attention: we run multiple attention operations in parallel, each with its own learned transformations.

Think of each head as a specialist looking for a particular type of relationship. Head 1 might learn to find sequence neighbors. Head 2 might learn to identify potential interaction partners based on amino acid chemistry. Head 3 might capture patterns reminiscent of secondary structure. Head 4 might discover functional relationships.

Mathematically, we have $h$ different attention heads, each with its own projection matrices:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

We then concatenate the outputs from all heads and project them back to the original dimension:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Here is a self-attention implementation that brings these concepts together:

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch, num_heads, seq_len, head_dim)

        # Compute attention
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )

        # Final projection
        output = self.out_proj(attn_output)

        return output, attn_weights
```

---

## Why Transformers Revolutionized Natural Language Processing

Before applying transformers to proteins, let us understand why they transformed natural language processing so dramatically. This context helps us appreciate what makes transformers special and why they transfer so well to biological sequences.

In 2017, researchers at Google published a paper with the provocative title "Attention Is All You Need." Their key insight was that the recurrent processing of RNNs was not fundamentally necessary. By relying entirely on attention mechanisms, they could build models that were not only more parallelizable but also more effective at capturing long-range dependencies.

The transformer architecture achieved state-of-the-art results on machine translation and quickly spread to virtually every language task. Models like BERT, GPT, and their successors built on the transformer foundation, culminating in the large language models that power today's AI assistants.

Why did transformers work so well for language? Natural language has long-range dependencies: the meaning of a pronoun depends on its antecedent, which might be many sentences earlier. The interpretation of a word depends on context that might span an entire document. Attention mechanisms can capture these dependencies directly, without the information bottleneck of sequential processing.

Protein sequences share these characteristics. A cysteine's functional role depends on whether it has a disulfide bonding partner hundreds of positions away. An active site's catalytic power depends on the precise arrangement of residues scattered throughout the sequence. The evolutionary conservation of a position depends on its structural and functional relationships with other positions. Transformers are natural candidates for modeling these long-range dependencies in protein sequences.

---

## The Complete Transformer Architecture

A transformer is more than just attention. It combines several components into a powerful architecture.

The basic building block is the transformer block, which consists of:

1. **Multi-head self-attention**: Each position attends to all positions, capturing pairwise relationships.

2. **Layer normalization**: A technique that normalizes the inputs to each sub-layer, stabilizing training.

3. **Feed-forward network**: A two-layer MLP applied independently to each position, providing the model with non-linear transformation capacity.

4. **Residual connections**: Skip connections that add the input of each sub-layer to its output, facilitating gradient flow and allowing the model to preserve information.

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x
```

The transformer processes sequences by stacking multiple such blocks. Information flows up through the layers, with each layer refining the representations based on increasingly complex patterns.

---

## Giving Transformers a Sense of Position

There is a subtle but critical issue with the attention mechanism as we have described it: it has no sense of position. If we shuffle the input positions randomly and apply attention, the computation is identical except for the shuffling. The attention weights depend only on the content of each position, not on where that position is in the sequence.

This is clearly problematic for proteins. Position matters enormously. Two glycines at positions 3 and 4 have a very different structural implication than glycines at positions 3 and 300. The backbone connectivity of the protein chain imposes constraints that depend critically on sequence position.

The solution is to inject positional information directly into the input representations. There are several approaches.

**Sinusoidal positional encoding** was the original approach from "Attention Is All You Need." It adds a fixed pattern of sine and cosine waves at different frequencies:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

The clever design of these encodings means that relative positions can be computed from the encodings themselves, and the model can generalize to sequence lengths longer than those seen during training.

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        return x + self.pe[:, :x.size(1)]
```

**Learned positional embeddings** take a simpler approach: treat positions like a vocabulary and learn an embedding vector for each position. This is more flexible but cannot generalize beyond the maximum training length.

**Rotary Position Embedding (RoPE)** is a more recent innovation used in models like ESM-2. Instead of adding position information to the embeddings, it encodes position in the rotation of key and query vectors. This approach elegantly handles relative positions and has become standard in modern protein language models.

---

## Protein Language Models: ESM and the Evolutionary Scale

The real payoff of transformers for proteins came with the development of protein language models. The flagship example is ESM (Evolutionary Scale Modeling), developed at Meta AI.

The core idea is remarkably simple: train a transformer on millions of protein sequences using masked language modeling. Randomly mask 15% of the amino acids in each sequence, then train the model to predict what amino acids should fill those masked positions. This is the same approach that made BERT so successful for natural language.

```python
class ProteinBERT(nn.Module):
    def __init__(self, vocab_size=33, embed_dim=768, num_heads=12, num_layers=12):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size, embed_dim, num_heads, num_layers
        )

        # MLM head
        self.mlm_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size)
        )

    def forward(self, x, mask=None):
        hidden = self.encoder(x, mask)
        logits = self.mlm_head(hidden)
        return logits, hidden
```

Why does this work so well for proteins? To predict a masked amino acid, the model must understand the context: what other amino acids are nearby, what structural constraints exist, what functional requirements the protein must satisfy. The model implicitly learns biochemistry, protein folding principles, and evolutionary relationships, all from the simple objective of filling in the blanks.

ESM-2, the most recent iteration, was trained on hundreds of millions of protein sequences. The resulting embeddings capture rich biological information:

```python
# Using ESM-2 for protein embeddings
import esm

# Load model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

# Prepare data
sequences = [("protein1", "MKTAYIAKQRQISFVKSHFSRQLE")]
batch_labels, batch_strs, batch_tokens = batch_converter(sequences)

# Get embeddings
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33])
    embeddings = results["representations"][33]  # (batch, seq_len, 1280)
```

These embeddings have been shown to encode:
- **Structural information**: Residues that are close in 3D space have similar embeddings, even if they are far apart in sequence.
- **Functional information**: Catalytic residues, binding sites, and post-translational modification sites are identifiable from their embeddings.
- **Evolutionary relationships**: The attention patterns learned by ESM recapitulate the co-evolutionary signals that biologists have studied for decades.

Most remarkably, the attention weights themselves can be used to predict protein contact maps with surprising accuracy, connecting back to our opening discussion of co-evolution.

---

## Proteins as Graphs: A Natural Representation

Now let us shift our perspective. We have been treating proteins as sequences, linear chains of amino acids. But proteins are not really linear objects. They fold into intricate three-dimensional structures where residues that are far apart in sequence come into close spatial contact.

A protein structure is naturally described as a graph. The nodes are residues (or atoms, if you want finer resolution). The edges represent relationships between residues: covalent bonds, spatial proximity, or any other interaction you want to capture.

This graph representation has several advantages:

**It captures 3D structure directly.** Instead of hoping that a sequence model will somehow learn about spatial relationships, we encode them explicitly in the graph connectivity.

**It handles variable size naturally.** Proteins come in all sizes, from small peptides to massive complexes. Graphs accommodate any number of nodes without the fixed-size constraints of some other architectures.

**It can encode rich relational information.** Edges can carry features describing the type and strength of interactions. You can have different edge types for backbone bonds, hydrogen bonds, salt bridges, and hydrophobic contacts.

Let us see how to convert a protein structure into a graph:

```python
import torch_geometric
from torch_geometric.data import Data

def protein_to_graph(coords, sequence, k=10, threshold=10.0):
    """
    Convert protein to graph.

    Args:
        coords: (N, 3) CA coordinates
        sequence: string of amino acids
        k: number of nearest neighbors
        threshold: distance threshold for edges

    Returns:
        PyG Data object
    """
    N = len(sequence)

    # Node features: one-hot amino acid
    aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    x = torch.zeros(N, 20)
    for i, aa in enumerate(sequence):
        if aa in aa_to_idx:
            x[i, aa_to_idx[aa]] = 1.0

    # Compute distances
    dist = torch.cdist(torch.tensor(coords), torch.tensor(coords))

    # Create edges (k-NN or threshold)
    edge_index = []
    edge_attr = []

    for i in range(N):
        # Get k nearest neighbors
        _, neighbors = dist[i].topk(k + 1, largest=False)
        neighbors = neighbors[1:]  # Exclude self

        for j in neighbors:
            if dist[i, j] < threshold:
                edge_index.append([i, j.item()])
                edge_attr.append([dist[i, j].item()])

    edge_index = torch.tensor(edge_index).T
    edge_attr = torch.tensor(edge_attr)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=coords)
```

---

## Message Passing: The Unifying Framework

All graph neural networks share a common computational pattern called message passing. The intuition is simple: each node gathers information from its neighbors, combines this information, and updates its own representation.

Imagine you are a residue in a protein. You want to update your understanding of yourself based on your structural neighborhood. You send a "message" to each of your neighbors asking about their state. They respond with information about themselves. You aggregate all these messages, combine them with your current state, and compute a new representation.

Mathematically, a message passing layer computes:

$$h_i^{(l+1)} = \phi\left(h_i^{(l)}, \bigoplus_{j \in \mathcal{N}(i)} \psi(h_i^{(l)}, h_j^{(l)}, e_{ij})\right)$$

Where:
- $h_i^{(l)}$ is the representation of node $i$ at layer $l$
- $\mathcal{N}(i)$ is the set of neighbors of node $i$
- $\psi$ is the message function, computing what information to send
- $\bigoplus$ is the aggregation function (sum, mean, or max)
- $\phi$ is the update function, combining the node's current state with the aggregated messages

Different GNN architectures make different choices for these functions, but they all fit within this framework.

---

## Graph Convolutional Networks: The Foundation

The Graph Convolutional Network (GCN) by Kipf and Welling in 2017 established the modern GNN paradigm. The GCN layer is elegantly simple:

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)}\right)$$

This equation looks intimidating, but it has a clear interpretation. The matrix $\tilde{A} = A + I$ is the adjacency matrix with added self-loops, so each node is also connected to itself. The matrix $\tilde{D}$ contains the degree of each node. The normalization $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ ensures that the aggregation is properly scaled: nodes with many neighbors do not dominate the computation.

In effect, each GCN layer computes a weighted average of each node's own features and its neighbors' features, then applies a linear transformation and nonlinearity.

```python
import torch_geometric.nn as gnn

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3):
        super().__init__()

        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(gnn.GCNConv(in_dim, hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(gnn.GCNConv(hidden_dim, hidden_dim))

        # Output layer
        self.layers.append(gnn.GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)

        x = self.layers[-1](x, edge_index)
        return x
```

The GCN treats all neighbors equally, which is both a strength and a limitation. It is simple and computationally efficient, but it cannot distinguish between more or less important neighbors.

---

## Graph Attention Networks: Learning Which Neighbors Matter

Just as attention transformed sequence modeling, it can transform graph neural networks. The Graph Attention Network (GAT) learns to weight neighbors differently based on their relevance.

The core idea is to compute attention coefficients between each node and its neighbors:

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T[W\mathbf{h}_i \| W\mathbf{h}_j]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(\mathbf{a}^T[W\mathbf{h}_i \| W\mathbf{h}_k]))}$$

This attention coefficient $\alpha_{ij}$ tells us how much node $i$ should pay attention to neighbor $j$. The learnable parameters $W$ and $\mathbf{a}$ allow the network to discover which neighbor relationships are most informative for the task at hand.

The updated node representation is then:

$$\mathbf{h}_i' = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} W\mathbf{h}_j\right)$$

For proteins, this is extremely useful. Not all spatial neighbors are equally important. A catalytic residue should pay more attention to other members of the active site than to random nearby residues. A residue in the hydrophobic core should focus on residues that contribute to the core's stability. GAT can learn these patterns.

```python
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4, num_layers=3):
        super().__init__()

        self.layers = nn.ModuleList()

        # First layer (concat heads)
        self.layers.append(gnn.GATConv(in_dim, hidden_dim, heads=num_heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(gnn.GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads))

        # Output layer (average heads)
        self.layers.append(gnn.GATConv(hidden_dim * num_heads, out_dim, heads=1, concat=False))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        x = self.layers[-1](x, edge_index)
        return x
```

---

## Message Passing Neural Networks: The General Framework

The Message Passing Neural Network (MPNN) framework provides maximum flexibility. Instead of using a fixed message and update rule, MPNN lets you define arbitrary neural networks for these functions.

**Message function:** Given the representations of two connected nodes and the edge between them, compute a message:

$$m_i^{(l+1)} = \sum_{j \in \mathcal{N}(i)} M(h_i^{(l)}, h_j^{(l)}, e_{ij})$$

**Update function:** Combine the node's current representation with the aggregated messages:

$$h_i^{(l+1)} = U(h_i^{(l)}, m_i^{(l+1)})$$

Both $M$ and $U$ can be arbitrary neural networks, typically MLPs. This flexibility allows MPNNs to learn complex, task-specific message passing schemes.

```python
class MPNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()

        # Message network
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Update network
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        N = x.size(0)

        # Compute messages
        message_input = torch.cat([x[row], x[col], edge_attr], dim=-1)
        messages = self.message_mlp(message_input)

        # Aggregate messages
        aggregated = torch.zeros(N, messages.size(-1), device=x.device)
        aggregated.scatter_add_(0, col.unsqueeze(-1).expand_as(messages), messages)

        # Update nodes
        update_input = torch.cat([x, aggregated], dim=-1)
        x = self.update_mlp(update_input)

        return x
```

For proteins, MPNNs can incorporate rich edge features: distances, angles, sequence separation, bond types. This makes them particularly well-suited for structure-based predictions.

---

## SE(3)-Equivariant Graph Neural Networks: Respecting Physical Symmetry

When we work with 3D protein structures, there is a crucial property we should respect: the laws of physics do not change if we rotate or translate the coordinate system. A protein's energy, its stability, its function are all independent of how we orient it in space.

This principle is called SE(3)-equivariance, where SE(3) is the mathematical group of rotations and translations in 3D space. An SE(3)-equivariant model produces outputs that transform appropriately under coordinate transformations:

- **Invariant outputs** (like energy or binding affinity) should not change at all when we rotate the protein.
- **Equivariant outputs** (like force vectors or coordinate updates) should rotate along with the protein.

Standard GNNs that operate on coordinates directly are not equivariant. If you rotate the input coordinates, the outputs do not rotate in a consistent way. This means the model must waste capacity learning the same function for every possible orientation, and it may generalize poorly to orientations not seen during training.

SE(3)-equivariant GNNs solve this problem by carefully designing the message passing operations to respect 3D symmetry. They operate on geometric features like relative positions and directions rather than absolute coordinates, and they use mathematical tools from representation theory to ensure that vector and tensor quantities transform correctly.

This is the architecture underlying many of the most impressive recent results in protein structure prediction and design, including key components of AlphaFold and protein diffusion models. While the full details involve sophisticated mathematics, the key insight is straightforward: by building the right symmetries into our models, we get better generalization with less data.

---

## Putting It All Together: A Protein Structure GNN

Let us build a complete GNN for making predictions about protein structures:

```python
class ProteinGNN(nn.Module):
    """GNN for per-residue predictions from structure."""

    def __init__(self, node_dim=20, edge_dim=32, hidden_dim=128, num_layers=5, num_classes=3):
        super().__init__()

        # Node embedding
        self.node_embed = nn.Linear(node_dim, hidden_dim)

        # Edge embedding (distance, sequence separation)
        self.edge_embed = nn.Sequential(
            nn.Linear(2, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim)
        )

        # MPNN layers
        self.layers = nn.ModuleList([
            MPNNLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # Output
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, pos):
        # x: (N, node_dim) node features
        # edge_index: (2, E) edges
        # pos: (N, 3) coordinates

        row, col = edge_index

        # Compute edge features
        distances = (pos[row] - pos[col]).norm(dim=-1, keepdim=True)
        seq_sep = torch.abs(row - col).float().unsqueeze(-1)
        edge_attr = self.edge_embed(torch.cat([distances, seq_sep / 100], dim=-1))

        # Embed nodes
        x = self.node_embed(x)

        # Message passing
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(x, edge_index, edge_attr)
            x = norm(x)

        # Classify
        logits = self.classifier(x)

        return logits
```

This model takes a protein structure as input, with amino acid identities as node features and 3D coordinates defining the spatial relationships. It computes edge features encoding both distance and sequence separation, then applies multiple rounds of message passing to let residues exchange information with their neighbors. Finally, it makes a prediction for each residue.

---

## The Connection to AlphaFold

Everything we have discussed in this lecture comes together in AlphaFold, perhaps the most celebrated application of deep learning to biology. AlphaFold combines:

- **Transformers over multiple sequence alignments**: The model processes evolutionary information using attention mechanisms that can identify co-evolving positions, exactly the intuition we started with.

- **Transformers over the pair representation**: AlphaFold maintains a representation of all pairwise relationships between residues, updated through transformer-like attention.

- **Graph neural networks for structure refinement**: The Invariant Point Attention (IPA) module in AlphaFold's structure module is a form of SE(3)-equivariant GNN, using the predicted 3D coordinates to define the graph structure.

The power of AlphaFold comes from this synergistic combination: transformers capture long-range dependencies in sequence and co-evolution, while equivariant GNNs ensure that the predicted structures respect physical symmetry.

Similarly, ESM-2 and ESMFold demonstrate that pure transformer models trained on sequences alone can learn to predict structure, showing that the attention mechanism can discover structural relationships without explicit 3D information.

---

## Choosing the Right Architecture

Given the variety of architectures we have covered, how do you choose the right one for your protein modeling task?

**For sequence-only tasks** (function prediction, subcellular localization, signal peptide detection):
- Start with pre-trained protein language models like ESM-2
- Fine-tune with task-specific heads
- Transformers excel at capturing long-range sequence patterns

**For structure-based tasks** (binding site prediction, stability prediction, function from structure):
- Use GNNs that take coordinates as input
- Consider SE(3)-equivariant architectures if you need to predict vectors or coordinate changes
- MPNNs with rich edge features capture the local geometric environment

**For sequence-structure joint modeling** (structure prediction, structure-conditioned design):
- Combine transformers for sequence processing with GNNs for structure reasoning
- This is the AlphaFold paradigm

**For very long proteins** (> 1000 residues):
- Standard transformer attention is O(L^2), which becomes prohibitive
- Use sparse attention, windowed attention, or efficient transformer variants
- GNNs scale better since they only consider local neighbors

---

## Key Takeaways

1. **Attention allows direct pairwise interactions** between all positions in a sequence, overcoming the sequential processing bottleneck of RNNs. This is crucial for capturing long-range dependencies in proteins.

2. **Queries, Keys, and Values** have intuitive interpretations: queries ask "what am I looking for?", keys advertise "what do I have?", and values carry the actual information that gets transmitted.

3. **Multi-head attention** lets the model capture different types of relationships simultaneously, with different heads specializing in different patterns.

4. **Transformers** combine attention with feed-forward networks, layer normalization, and residual connections into a powerful architecture that has revolutionized both NLP and computational biology.

5. **Protein language models** like ESM learn rich representations by predicting masked amino acids, implicitly capturing biochemistry, structure, and evolution.

6. **Graph neural networks** represent proteins as graphs, naturally encoding 3D structure. The message passing framework unifies different GNN architectures.

7. **GAT learns attention over graphs**, allowing the model to weight neighbors by importance rather than treating them equally.

8. **SE(3)-equivariant GNNs** respect the symmetries of 3D space, ensuring that predictions transform correctly under rotation and translation.

9. **AlphaFold combines transformers and equivariant GNNs**, showing that the architectures we have studied are complementary and most powerful when used together.

---

## Exercises

1. Implement scaled dot-product attention from scratch and visualize the attention patterns learned by a simple protein transformer. Which positions attend to which other positions?

2. Build a transformer-based classifier for protein family prediction. Compare its performance to the RNN models from the previous lecture.

3. Compare GCN vs GAT on a protein function prediction task using structural data. Do the learned attention weights reveal biologically meaningful patterns?

4. Implement an MPNN with edge updates for contact prediction. How do the results compare to attention-based contact prediction from ESM?

---

## References

- Vaswani et al. (2017). "Attention Is All You Need" - The original transformer paper.

- Rives et al. (2021). "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences" - The ESM paper that brought transformers to proteins.

- Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks" - The foundational GCN paper.

- Velickovic et al. (2018). "Graph Attention Networks" - Introducing attention to GNNs.

- Gilmer et al. (2017). "Neural Message Passing for Quantum Chemistry" - The MPNN framework.

- Jumper et al. (2021). "Highly accurate protein structure prediction with AlphaFold" - The revolutionary structure prediction model that synthesizes these architectures.
