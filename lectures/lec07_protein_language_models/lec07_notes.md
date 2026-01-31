# Lecture 7: Protein Language Models

## The Deep Connection Between Evolution and Language

Evolution is nature's language model. For billions of years, the process of natural selection has been editing, refining, and optimizing protein sequences with the same ruthless efficiency that a skilled author revises a manuscript. Every amino acid in a protein has been tested against the harshest of critics: survival. The proteins we observe today are the survivors, the sentences that made sense, the grammatically correct statements in the language of life.

This profound insight lies at the heart of protein language models. When we train a neural network to predict missing amino acids in a protein sequence, we are not merely teaching it pattern recognition. We are asking it to learn the implicit grammar that evolution has enforced over eons. The model learns which amino acids can substitute for which others, which combinations are forbidden, and which patterns are universally conserved. In doing so, it absorbs the accumulated wisdom of billions of years of molecular experimentation.

The implications of this perspective are staggering. If proteins are indeed a language, then the hundreds of millions of protein sequences in databases like UniProt are not just data points. They are texts, a vast library written by evolution itself. And if we can build a model that truly understands this language, we gain the ability to read, interpret, and perhaps even write new entries in this ancient library.

## Why Proteins Are Like Language

The analogy between proteins and language is more than a convenient metaphor. It reflects deep structural similarities that make natural language processing techniques remarkably effective for protein analysis.

Consider how human language works. Words combine to form sentences according to grammatical rules. Some word combinations are meaningful while others are nonsensical. The meaning of a word often depends on its context. Synonyms exist, words that differ in form but serve similar functions. And importantly, language evolves over time while maintaining core structural principles.

Proteins exhibit all of these characteristics. Amino acids are the words of protein language, twenty distinct chemical entities that serve as the building blocks. The protein sequence is a sentence, a linear string of these words that encodes a specific meaning, which in this case means a three-dimensional structure and biological function. Just as grammatical rules constrain which word sequences are valid sentences, biochemical and physical constraints determine which amino acid sequences can fold into functional proteins.

The grammar of proteins is extraordinarily strict. Most random sequences of amino acids will not fold into stable structures. They will aggregate, misfold, or simply fail to perform any useful function. This is analogous to the observation that most random strings of English letters do not form valid sentences. The sequences we observe in nature have passed through the filter of natural selection, which only allows grammatically correct protein sentences to survive.

Context matters just as much in proteins as it does in language. The same amino acid can play very different roles depending on its neighbors. A hydrophobic residue buried in the protein core contributes to stability, while the same residue on the surface might create a binding site for another molecule. Understanding these context-dependent meanings is precisely what language models excel at.

Perhaps most importantly, proteins exhibit synonymy and semantic similarity. Different amino acids can sometimes substitute for each other without destroying function, just as different words can convey similar meanings. A leucine can often replace an isoleucine because they have similar physical properties. This is not random; it reflects the underlying biochemistry, the semantics of the protein language.

The table below summarizes these correspondences:

| Natural Language | Protein Language |
|-----------------|------------------|
| Words/tokens | Amino acids |
| Sentences | Protein sequences |
| Grammar rules | Biochemical constraints |
| Semantics | Structure and function |
| Synonyms | Functionally similar amino acids |
| Co-occurrence patterns | Co-evolution of residues |

This analogy suggests a tantalizing possibility: perhaps the same machine learning techniques that have revolutionized natural language processing can be applied to proteins. The answer, as we will see, is a resounding yes.

## Learning from Billions of Evolutionary Experiments

Why should we expect language models to work for proteins? The answer lies in understanding what these models actually learn.

When we train a model to predict masked amino acids, we are implicitly teaching it about evolutionary constraints. Consider what happens when a model sees thousands of sequences of a particular protein family. It observes which positions are absolutely conserved, never changing across millions of years of evolution. It notices which positions vary but only within certain classes of amino acids. It detects patterns of co-variation, positions that change together because they are in physical contact in the folded structure.

All of this information is encoded in the statistical patterns of sequences. The model does not need to be told about protein structure or function explicitly. It discovers these concepts by learning to predict sequences accurately. This is the magic of self-supervised learning applied to biology.

The scale of available data makes this approach particularly powerful. The UniProt database contains hundreds of millions of protein sequences, and metagenomic surveys are adding billions more. This is far more training data than exists for most natural language tasks. Each of these sequences represents a successful evolutionary experiment, a design that works well enough to have been preserved by natural selection.

By learning from this vast corpus, protein language models capture evolutionary knowledge that would be impossible to encode manually. They learn the subtle rules that determine whether a sequence will fold properly. They internalize the constraints that govern enzyme active sites, membrane-spanning regions, and protein-protein interfaces. They absorb the accumulated wisdom of four billion years of molecular evolution.

## Masked Language Modeling: A Simple but Powerful Idea

The dominant approach for training protein language models is masked language modeling, or MLM. The concept is elegantly simple: hide some amino acids, then ask the model to predict them back.

In practice, we randomly select about 15% of positions in a protein sequence and replace them with a special mask token. The model sees the corrupted sequence and must guess the identity of the hidden amino acids. To do this successfully, the model must learn to use the context, the surrounding amino acids, to infer what was hidden.

This is remarkably similar to a fill-in-the-blank exercise. If I show you the sentence "The cat sat on the ___", you can easily guess that the missing word is probably "mat" or "floor" or "chair". You make this prediction based on your understanding of English grammar and semantics. The masked language model is doing the same thing with proteins.

The mathematical formulation is straightforward. Let x be a protein sequence and M be the set of masked positions. The training objective is to maximize:

$$\mathcal{L}_{MLM} = -\mathbb{E}_{x \sim \mathcal{D}} \left[ \sum_{i \in \mathcal{M}} \log p_\theta(x_i | x_{\backslash \mathcal{M}}) \right]$$

In words, we want to maximize the probability of the correct amino acid at each masked position, given all the unmasked positions. The model learns parameters that make correct predictions more likely.

What makes this approach so powerful is that it requires no labels. We do not need experimentally measured properties or human annotations. The sequences themselves provide the supervision. This is crucial because while we have hundreds of millions of protein sequences, we have experimental data for only a tiny fraction of them.

The masking strategy follows a specific protocol borrowed from the BERT model in NLP. When a position is selected for masking, 80% of the time it receives the mask token. 10% of the time it is replaced with a random amino acid. And 10% of the time it is left unchanged. This variety helps the model learn more robust representations.

```python
import torch
import torch.nn as nn

def mask_tokens(sequence, vocab_size, mask_token_id, mask_prob=0.15):
    """
    Apply MLM masking strategy to a protein sequence.

    The masking follows BERT's strategy: 80% [MASK], 10% random, 10% original.
    This helps the model learn robust representations.
    """
    labels = sequence.clone()

    # Randomly select positions to mask
    probability_matrix = torch.full(sequence.shape, mask_prob)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Only compute loss on masked positions
    labels[~masked_indices] = -100

    # 80% of the time, replace with [MASK]
    indices_replaced = torch.bernoulli(
        torch.full(sequence.shape, 0.8)
    ).bool() & masked_indices
    sequence[indices_replaced] = mask_token_id

    # 10% of the time, replace with random token
    indices_random = torch.bernoulli(
        torch.full(sequence.shape, 0.5)
    ).bool() & masked_indices & ~indices_replaced
    random_tokens = torch.randint(vocab_size, sequence.shape)
    sequence[indices_random] = random_tokens[indices_random]

    # 10% of the time, keep original
    return sequence, labels
```

There is an alternative approach called autoregressive modeling, used by models like ProGen and ProtGPT2. Instead of predicting masked tokens, autoregressive models predict sequences left-to-right, one amino acid at a time. Each prediction uses only the preceding context. This is natural for generation tasks but less effective for creating bidirectional representations useful in understanding tasks.

## ESM-2: The State of the Art

ESM-2, short for Evolutionary Scale Modeling 2, represents the current state of the art in protein language models. Developed by researchers at Meta AI, ESM-2 comes in a range of sizes from 8 million to 15 billion parameters. The largest models achieve remarkable performance across a wide variety of protein understanding tasks.

The ESM-2 family was trained on UniRef50, a clustered version of the UniProt database containing roughly 60 million representative protein sequences. Training the largest models required thousands of GPUs running for weeks, an investment that would be impractical for most research groups but that pays dividends when the resulting models are made freely available.

What can ESM-2 do that was not possible before? The answer touches nearly every aspect of computational protein science.

First, ESM-2 provides rich sequence embeddings. For any protein sequence, the model produces a numerical representation that captures evolutionary and structural information. These embeddings can be used as features for downstream machine learning tasks, often outperforming hand-crafted features developed over decades.

Second, ESM-2 enables zero-shot prediction of mutational effects. Without any fine-tuning, the model can estimate whether a particular mutation is likely to be beneficial, neutral, or deleterious. This capability emerges directly from learning the evolutionary distribution of sequences.

Third, ESM-2 representations improve performance on virtually every protein property prediction task that has been tested. Secondary structure prediction, subcellular localization, function annotation, and contact prediction all benefit from ESM-2 embeddings.

Fourth, and most remarkably, ESM-2 representations contain enough information to predict three-dimensional structure directly. The ESMFold model, built on ESM-2 embeddings, achieves structure prediction accuracy competitive with AlphaFold2 while running much faster because it does not require multiple sequence alignments.

The different model sizes offer tradeoffs between performance and computational requirements:

| Model | Parameters | Layers | Embedding Dim | Use Case |
|-------|-----------|--------|---------------|----------|
| ESM-2 8M | 8M | 6 | 320 | Quick prototyping |
| ESM-2 35M | 35M | 12 | 480 | Lightweight applications |
| ESM-2 150M | 150M | 30 | 640 | Good balance |
| ESM-2 650M | 650M | 33 | 1280 | High performance |
| ESM-2 3B | 3B | 36 | 2560 | Near-state-of-art |
| ESM-2 15B | 15B | 48 | 5120 | Maximum performance |

For most applications, the 650M parameter model offers an excellent balance between performance and computational cost. It runs comfortably on a single GPU and provides embeddings that are nearly as informative as those from the largest models.

## Inside the Transformer Architecture

ESM-2 is built on the Transformer architecture, the same foundation that powers large language models like GPT and BERT. Understanding how Transformers work illuminates why they are so effective for proteins.

The core innovation of Transformers is the attention mechanism. Attention allows each position in a sequence to directly interact with every other position, regardless of their distance in the sequence. This is crucial for proteins, where amino acids that are far apart in sequence may be close together in the folded structure.

Consider a protein with 200 amino acids. In the folded structure, residue 10 might be in direct contact with residue 150. A model that only looks at local context would miss this relationship entirely. But attention allows residue 10 to attend to residue 150, learning patterns that span the entire protein.

The attention computation works as follows. Each position is projected into three vectors: a query, a key, and a value. The query represents what information this position is looking for. The key represents what information this position has to offer. The value represents the actual content to be aggregated.

Attention weights are computed by taking the dot product between queries and keys, then applying a softmax to get a probability distribution. These weights determine how much each position attends to each other position. The output is a weighted sum of values.

Mathematically:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

ESM-2 uses several architectural refinements that improve performance:

**Rotary Position Embeddings (RoPE)** encode relative positions directly into the attention computation. Instead of adding position embeddings to the input, RoPE rotates the query and key vectors based on their positions. This helps the model generalize to sequences longer than those seen during training.

**Pre-Layer Normalization** applies layer normalization before the attention and feedforward operations rather than after. This simple change makes training more stable, especially for very deep models.

**SwiGLU Activation** replaces the standard ReLU or GELU activation in the feedforward layers. SwiGLU combines the Swish activation with a gating mechanism, providing more expressive transformations.

```python
class SwiGLU(nn.Module):
    """
    SwiGLU activation for Transformer feedforward layers.
    Combines the Swish activation with a gating mechanism.
    """
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.w3(nn.functional.silu(self.w1(x)) * self.w2(x))
```

These architectural choices, combined with massive scale training, produce representations that capture remarkably rich information about proteins.

## Extracting and Using Embeddings

One of the most practical applications of ESM-2 is extracting embeddings for use in downstream tasks. An embedding is a numerical vector that represents a protein, or individual positions within a protein, in a way that captures relevant biological information.

Getting started with ESM-2 embeddings is straightforward:

```python
import torch
import esm

def extract_esm_embeddings(sequences, model_name="esm2_t33_650M_UR50D"):
    """
    Extract ESM-2 embeddings for protein sequences.

    Returns per-residue embeddings and a mean embedding for each sequence.
    """
    # Load model
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prepare batch
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_tokens = batch_tokens.to(device)

    # Extract embeddings from the last layer
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    # Process results
    token_embeddings = results["representations"][33]

    embeddings = {}
    for i, (label, seq) in enumerate(sequences):
        seq_len = len(seq)
        embeddings[label] = {
            'per_residue': token_embeddings[i, 1:seq_len+1].cpu(),  # [L, 1280]
            'mean': token_embeddings[i, 1:seq_len+1].mean(0).cpu(),  # [1280]
        }

    return embeddings
```

The per-residue embeddings have dimension equal to the model's hidden size, 1280 for the 650M model. Each position receives a vector that encodes information about that residue in its sequence context. These per-residue embeddings are useful for tasks that require position-specific predictions, such as secondary structure or binding site identification.

For tasks that require a single vector representing the entire protein, we need a pooling strategy. The simplest approach is mean pooling, averaging the embeddings across all positions. Alternatives include using the embedding of a special beginning-of-sequence token, or attention-weighted pooling that gives more weight to important positions.

What information is actually captured in these embeddings? Research has shown that ESM embeddings encode:

- **Secondary structure**: Positions in helices, sheets, and loops occupy different regions of embedding space
- **Solvent accessibility**: Buried and exposed residues are distinguishable
- **Functional sites**: Active sites and binding pockets have characteristic embeddings
- **Evolutionary conservation**: Highly conserved positions have distinct signatures
- **Structural contacts**: Positions that are close in 3D space have related embeddings

This rich information makes ESM embeddings excellent features for training downstream models. Instead of engineering features by hand, you can simply extract embeddings and train a small neural network on top.

## Zero-Shot Mutation Prediction: Magic from Statistics

Perhaps the most remarkable capability of protein language models is zero-shot prediction of mutational effects. Without any fine-tuning or task-specific training, ESM-2 can estimate whether a particular amino acid substitution is likely to be harmful.

How does this work? The key insight is that the model has learned the distribution of natural sequences. Mutations that push a sequence away from this learned distribution are likely to be deleterious, while mutations that stay within the distribution are more likely to be tolerated.

Concretely, we can score a mutation by comparing the model's predicted probability of the mutant amino acid versus the wild-type amino acid at that position. The procedure is:

1. Take the wild-type sequence and mask the position of interest
2. Run the model to get predicted probabilities for each amino acid
3. Compare log P(mutant) to log P(wild-type)
4. The log-likelihood ratio indicates the predicted effect

```python
def predict_mutation_effect(sequence, position, original_aa, mutant_aa,
                           model, alphabet):
    """
    Predict effect of mutation using log-likelihood ratio.

    A positive score suggests the mutation may be favorable.
    A negative score suggests the mutation may be deleterious.
    """
    batch_converter = alphabet.get_batch_converter()

    # Create masked sequence
    seq_list = list(sequence)
    seq_list[position] = '<mask>'
    masked_seq = ''.join(seq_list)

    # Get model predictions
    _, _, tokens = batch_converter([("seq", masked_seq)])

    with torch.no_grad():
        logits = model(tokens)["logits"]

    # Get log probabilities at the masked position
    log_probs = torch.log_softmax(logits[0, position + 1], dim=-1)

    # Look up probabilities for wild-type and mutant
    wt_idx = alphabet.get_idx(original_aa)
    mt_idx = alphabet.get_idx(mutant_aa)

    # Return log-likelihood ratio
    return (log_probs[mt_idx] - log_probs[wt_idx]).item()
```

Why does this work so well? The model has learned from hundreds of millions of evolutionary experiments. It knows which amino acids appear at which positions in functional proteins. When we ask whether a mutation is likely to be harmful, we are really asking: is this mutation consistent with what evolution has allowed?

Mutations to amino acids that never appear at a given position across all known homologs receive low probability scores. Mutations to amino acids that are common at that position receive high probability scores. This captures the essence of evolutionary constraint.

Remarkably, these zero-shot predictions correlate well with experimental measurements of mutational effects. Studies have shown that ESM-2 scores achieve state-of-the-art performance on benchmark datasets, often outperforming methods specifically trained on experimental data.

This capability has profound practical implications. Experimental measurement of mutational effects is slow and expensive, requiring deep mutational scanning experiments that can take months and cost hundreds of thousands of dollars. Zero-shot prediction provides an instant, free estimate that can guide experimental efforts and accelerate protein engineering.

## Fine-Tuning for Specific Tasks

While zero-shot capabilities are impressive, fine-tuning ESM-2 on task-specific data can further improve performance. Fine-tuning adapts the pre-trained representations to a specific problem, learning which aspects of the embeddings are most relevant.

The standard approach adds a small prediction head on top of the frozen or trainable ESM-2 backbone. For sequence classification tasks like predicting subcellular localization, this might be a simple feedforward network that takes the mean pooled embedding as input:

```python
class ESMClassifier(nn.Module):
    """
    ESM-2 with a classification head for sequence-level prediction.
    """
    def __init__(self, esm_model, hidden_dim=1280, num_classes=10, dropout=0.1):
        super().__init__()
        self.esm = esm_model
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, tokens):
        # Get ESM embeddings
        outputs = self.esm(tokens, repr_layers=[33])
        embeddings = outputs["representations"][33]

        # Mean pooling (excluding special tokens)
        mask = (tokens != 0) & (tokens != 1) & (tokens != 2)
        mask = mask.unsqueeze(-1).float()
        pooled = (embeddings * mask).sum(1) / mask.sum(1)

        # Classify
        return self.classifier(self.dropout(pooled))
```

For per-residue tasks like secondary structure prediction, each position gets its own prediction:

```python
class ESMTokenClassifier(nn.Module):
    """
    ESM-2 for per-residue prediction tasks.
    """
    def __init__(self, esm_model, hidden_dim=1280, num_labels=3):
        super().__init__()
        self.esm = esm_model

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_labels)
        )

    def forward(self, tokens):
        outputs = self.esm(tokens, repr_layers=[33])
        embeddings = outputs["representations"][33]
        return self.classifier(embeddings)
```

Full fine-tuning updates all parameters in the model, allowing the representations themselves to adapt to the task. This typically gives the best performance but requires significant computational resources and risks catastrophic forgetting, where the model loses its general-purpose capabilities.

## LoRA: Efficient Adaptation for Everyone

Full fine-tuning of a 650M parameter model requires substantial GPU memory and computational time. For the 3B or 15B parameter versions, it becomes prohibitive for most researchers. This is where Low-Rank Adaptation, or LoRA, comes to the rescue.

LoRA addresses a fundamental insight about fine-tuning: the weight updates are typically low-rank. Instead of modifying all 650 million parameters, we can achieve similar results by learning a small number of parameters that modulate the frozen pre-trained weights.

The mathematical formulation is elegant. For a pre-trained weight matrix W with dimensions d by k, we learn two small matrices: B with dimensions d by r, and A with dimensions r by k. The rank r is typically 8 or 16, vastly smaller than the original dimensions.

$$W_{adapted} = W_{original} + BA$$

During forward pass, the output is:
$$h = W_{original} \cdot x + \frac{\alpha}{r} \cdot BA \cdot x$$

The scaling factor alpha/r controls the magnitude of the adaptation. This formulation has several advantages:

**Massive parameter reduction**: For r=8 and a 1280x1280 weight matrix, we go from 1.6 million parameters to about 20,000. The total number of trainable parameters drops to less than 1% of the original model.

**Memory efficiency**: We only need to store optimizer states and gradients for the small LoRA matrices, not the full model weights.

**No catastrophic forgetting**: The original weights are frozen, preserving the model's general capabilities.

**Easy switching**: Different LoRA adaptations can be swapped in and out without reloading the full model, enabling efficient multi-task deployment.

```python
class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer that wraps a frozen linear layer.
    """
    def __init__(self, original_layer, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False

        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.dropout = nn.Dropout(dropout)

        # Standard initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.scaling = alpha / r

    def forward(self, x):
        # Original frozen forward pass
        result = self.original_layer(x)

        # Add LoRA contribution
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + lora_output * self.scaling
```

In practice, LoRA is typically applied to the query and value projection matrices in the attention layers. These have been empirically found to benefit most from adaptation. The PEFT library from Hugging Face makes applying LoRA straightforward:

```python
from transformers import EsmForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

def create_lora_model(model_name, num_labels, r=8, alpha=16):
    """
    Create an ESM model with LoRA adapters using the PEFT library.
    """
    # Load base model
    model = EsmForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
        bias="none"
    )

    # Create PEFT model
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model
```

When you run this code, you will see output showing that only a small fraction of parameters are trainable. For ESM-2 650M with r=8, this is typically around 0.1-0.5% of total parameters.

LoRA makes it possible to fine-tune state-of-the-art protein language models on a single consumer GPU. What was once the exclusive domain of large research labs is now accessible to any researcher with a laptop.

## From Embeddings to Structure: ESMFold

The ultimate test of whether a language model truly understands proteins is whether it can predict structure. The ESMFold model demonstrates that ESM-2 embeddings contain sufficient information to predict three-dimensional atomic coordinates.

ESMFold takes a protein sequence, passes it through ESM-2 to get embeddings, and then uses a structure prediction module to generate atomic coordinates. Unlike AlphaFold2, ESMFold does not require multiple sequence alignments. It predicts directly from a single sequence.

```python
import esm

def predict_structure(sequence):
    """
    Predict protein structure using ESMFold.
    Returns PDB-formatted structure.
    """
    model = esm.pretrained.esmfold_v1()
    model = model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        output = model.infer_pdb(sequence)

    return output
```

ESMFold achieves accuracy competitive with AlphaFold2 while running significantly faster. For a typical protein domain, ESMFold produces a structure in seconds rather than minutes. This speed advantage makes ESMFold particularly useful for large-scale structural analysis.

The success of ESMFold demonstrates something profound about what language models learn. The statistical patterns in sequence data encode sufficient information to recover three-dimensional structure. The grammar of proteins is intimately tied to their physics.

## Attention Maps as Windows into Protein Structure

The attention mechanism provides not just a technical advantage but also an interpretable window into what the model has learned. Each attention head learns to attend to different types of relationships between positions.

Remarkably, the attention patterns in ESM-2 correlate with residue-residue contacts in the three-dimensional structure. Positions that are close in space tend to attend strongly to each other, even if they are far apart in sequence. This suggests that the model has learned structural relationships purely from sequence statistics.

```python
def extract_attention_maps(model, tokens, layer=-1):
    """
    Extract attention maps from ESM model.

    Returns attention matrix of shape [heads, length, length]
    """
    with torch.no_grad():
        outputs = model(tokens, return_contacts=True)

    attentions = outputs["attentions"]
    return attentions[layer][0]  # First sequence in batch
```

We can use these attention maps for contact prediction by symmetrizing and applying a correction called APC, Average Product Correction, that removes background signal:

```python
def attention_to_contacts(attention, threshold=0.5):
    """
    Convert attention to predicted contact map.
    """
    # Average over attention heads
    attn_mean = attention.mean(0)

    # Symmetrize
    attn_sym = (attn_mean + attn_mean.T) / 2

    # APC correction
    row_mean = attn_sym.mean(1, keepdim=True)
    col_mean = attn_sym.mean(0, keepdim=True)
    overall_mean = attn_sym.mean()
    apc = (row_mean * col_mean) / overall_mean
    corrected = attn_sym - apc

    # Threshold to get binary contacts
    return (corrected > threshold).float()
```

This analysis reveals that the model has discovered the fundamental principle connecting sequence and structure: residues that co-evolve tend to be in contact. This principle, known from evolutionary coupling analysis, emerges automatically from masked language modeling.

## The Broader Landscape of Protein Language Models

ESM-2 is not the only protein language model. Several alternatives exist, each with different strengths:

**ProtTrans** from the Technical University of Munich offers multiple architectures including BERT, Albert, and T5-style models. The ProtT5 model has been particularly popular for its balance of performance and efficiency.

**ProGen** and **ProGen2** from Salesforce use autoregressive modeling, predicting sequences left-to-right. This makes them particularly effective for sequence generation and design tasks.

**Ankh** offers efficient training strategies that achieve competitive performance with less compute, making pre-training more accessible.

**ProteinBERT** from Google incorporates Gene Ontology annotations during pre-training, explicitly learning function alongside sequence patterns.

Each model has its niche, but ESM-2 remains the most widely used for general-purpose embedding extraction due to its strong performance across diverse tasks and the availability of multiple model sizes.

## Practical Considerations

When applying protein language models in practice, several considerations guide model selection and usage.

**Model size tradeoffs**: Larger models generally perform better but require more memory and compute. For many tasks, the 650M model offers the best balance. For quick experimentation, the 35M or 150M models run quickly on CPU.

**Sequence length limits**: ESM-2 was trained on sequences up to 1024 amino acids. Longer sequences require either truncation, chunking, or using models designed for longer contexts.

**Batch processing**: Processing many sequences efficiently requires attention to batching and padding. The ESM library handles this automatically, but custom pipelines need care.

**GPU memory**: The 650M model requires about 2.5GB of GPU memory for inference. Fine-tuning requires more. LoRA dramatically reduces memory requirements for fine-tuning.

**Reproducibility**: Set random seeds and use deterministic operations for reproducible results, especially important when comparing methods.

## Key Takeaways

Protein language models represent a paradigm shift in computational biology. By treating sequences as a language and learning from the vast corpus of natural proteins, these models capture evolutionary and structural knowledge that would be impossible to encode manually.

The masked language modeling objective is deceptively simple. Hide some amino acids, predict them back. Yet this simple task, when applied at scale, produces representations that encode secondary structure, function, evolutionary conservation, and even three-dimensional contacts.

ESM-2 is the current state of the art, offering models from 8 million to 15 billion parameters. The embeddings from these models improve performance on virtually every protein understanding task that has been tested.

Zero-shot mutation prediction emerges as a remarkable capability. Without any task-specific training, the model can estimate mutational effects by comparing how well mutations fit the learned distribution of natural sequences. This reflects the deep connection between evolution and the model's training objective.

LoRA enables efficient fine-tuning, reducing trainable parameters to less than 1% of the original model. This democratizes access to large-scale protein AI, allowing researchers with modest computational resources to adapt state-of-the-art models to their specific problems.

The attention patterns in these models correlate with residue contacts in three-dimensional structures, suggesting that the model has discovered the fundamental principle connecting sequence and structure. ESMFold takes this further, predicting atomic coordinates directly from ESM embeddings.

We are still in the early days of protein language models. As these models continue to scale and as new architectures emerge, their capabilities will only grow. The dream of reading and writing in the language of life is becoming reality.

---

## References

1. Rives, A. et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS*.
2. Lin, Z. et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*.
3. Hu, E.J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.
4. Elnaggar, A. et al. (2022). ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. *IEEE TPAMI*.
5. Meier, J. et al. (2021). Language models enable zero-shot prediction of the effects of mutations on protein function. *NeurIPS*.

---

## Exercises

1. **Embedding Visualization**: Extract ESM embeddings for members of a protein family (e.g., kinases or proteases). Use UMAP or t-SNE to visualize the embeddings in 2D. Do sequences with similar functions cluster together? What about sequences with similar structures?

2. **Zero-Shot Benchmark**: Choose a protein with available deep mutational scanning data. Use ESM-2 to predict the effect of all single mutations. Calculate the correlation between your predictions and experimental fitness values. How does performance vary with position in the protein?

3. **LoRA vs Full Fine-Tuning**: Take a small dataset of labeled proteins and compare two approaches: (a) full fine-tuning of ESM-2 35M, and (b) LoRA fine-tuning of ESM-2 650M. Compare performance, training time, and memory usage.

4. **Layer-wise Analysis**: Extract embeddings from multiple layers of ESM-2, not just the last layer. Train simple classifiers using embeddings from each layer. Which layers are most informative for different tasks?

5. **Attention Interpretation**: For a protein with known structure, extract attention maps and compare them to the true contact map. Which attention heads best predict contacts? Do different heads specialize for different types of interactions?
