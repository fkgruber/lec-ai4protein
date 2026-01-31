# Lecture 4: Training and Optimizing Neural Networks for Proteins

## Learning Objectives

By the end of this lecture, you will be able to:
1. Apply regularization techniques to prevent overfitting in protein models
2. Design learning rate schedules for stable and efficient training
3. Systematically tune hyperparameters for protein prediction tasks
4. Debug common neural network training failures
5. Address protein-specific challenges like variable-length sequences and data leakage

## Prerequisites

- Lecture 3: Introduction to AI and Deep Learning
- Understanding of PyTorch basics (tensors, nn.Module, training loops)
- Familiarity with gradient descent and backpropagation

---

## 1. Introduction: Why Training is the Hardest Part

When students first learn deep learning, they often focus on architectures: the elegance of attention mechanisms, the beauty of residual connections, the power of graph neural networks. But here is an uncomfortable truth that experienced practitioners know well: getting a neural network to train properly is usually harder than designing the network itself.

Consider this scenario. You have just implemented a sophisticated protein structure predictor, inspired by the latest AlphaFold paper. You run your first training experiment. The loss barely moves. You try a different learning rate. The loss explodes to infinity. You try yet another learning rate. The loss decreases for a while, then plateaus far above where it should be. What went wrong?

The answer could be almost anything. The learning rate might be wrong. The model might be memorizing the training data instead of learning generalizable patterns. The gradients might be vanishing in your deep network. There might be a subtle bug in your data loading. The batch normalization layers might be behaving unexpectedly during evaluation. The list goes on.

This lecture is about developing the intuition and toolkit to navigate these challenges. We will focus particularly on proteins, where the unique properties of biological data create their own training difficulties. But the principles we learn will apply broadly to any deep learning project.

Training neural networks is fundamentally an empirical science. Unlike traditional algorithm design where we can prove correctness, deep learning requires us to run experiments, observe behaviors, and iterate. The goal of this lecture is to make you a better experimentalist.

---

## 2. Regularization Techniques

### 2.1 The Overfitting Problem

Before we discuss solutions, let us understand the problem deeply. Overfitting occurs when a model learns patterns specific to the training data that do not generalize to new examples. In protein prediction, this is particularly insidious because proteins have complex, correlated features. A model might memorize that sequence pattern GAVL always appears in alpha helices in the training set, even if this is just a coincidence of the particular proteins sampled.

How do we detect overfitting? The classic signature is a gap between training and validation performance. If your training loss keeps decreasing while validation loss plateaus or increases, you are overfitting. But this is a trailing indicator. Ideally, we want to prevent overfitting proactively through regularization.

### 2.2 Dropout: Training an Ensemble

Dropout is perhaps the most widely used regularization technique, and its intuition is beautiful. During training, we randomly "drop" neurons by setting their outputs to zero with some probability p (typically 0.1 to 0.5). This has two effects.

First, it prevents co-adaptation. Without dropout, neurons can develop complex interdependencies: neuron A only fires when neuron B fires, which only happens when neuron C fires. These fragile chains break down on new data. Dropout forces each neuron to be useful on its own.

Second, dropout implicitly trains an exponential ensemble of networks. Each training step uses a different random subset of neurons, effectively training a different sub-network. At test time, we use all neurons (scaled appropriately), which approximates averaging predictions from all these sub-networks.

Here is how dropout looks in practice for a protein secondary structure predictor:

```python
import torch
import torch.nn as nn

class SecondaryStructurePredictor(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=256, num_classes=3, dropout_rate=0.3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout after activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Keep applying throughout
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.encoder(x)
```

A critical detail: dropout behaves differently during training and evaluation. You must call `model.train()` before training and `model.eval()` before evaluation. Forgetting this is a common source of bugs where validation performance looks artificially bad.

```python
# Training phase
model.train()
for batch in train_loader:
    outputs = model(batch)  # Dropout is active
    loss.backward()

# Evaluation phase
model.eval()
with torch.no_grad():
    for batch in val_loader:
        outputs = model(batch)  # Dropout is disabled, outputs scaled
```

### 2.3 Weight Decay: Keeping Parameters Small

Weight decay is L2 regularization applied to the loss function. Instead of minimizing just the task loss L, we minimize:

$$L_{total} = L_{task} + \lambda \sum_i w_i^2$$

The intuition is that large weights often indicate the model is fitting noise. By penalizing large weights, we encourage the model to find simpler solutions that rely on many small contributions rather than a few large ones.

In PyTorch, weight decay is built into the optimizer:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01  # Common values: 0.001 to 0.1
)
```

A subtle but important point: AdamW handles weight decay correctly, while Adam with weight_decay parameter implements L2 regularization differently (coupled with the adaptive learning rate). For most purposes, prefer AdamW.

For protein models, weight decay is particularly useful when working with limited training data. If you are fine-tuning a pre-trained language model like ESM on a small dataset of labeled proteins, weight decay helps prevent the model from drifting too far from its well-regularized pre-trained state.

### 2.4 Batch Normalization vs Layer Normalization

Normalization layers stabilize training by ensuring that layer inputs have reasonable distributions. Without normalization, small changes in early layers can cascade into large changes in later layers, making optimization difficult.

**Batch Normalization** normalizes across the batch dimension:

```python
# For each feature, compute mean and variance across the batch
x_norm = (x - mean(x, dim=batch)) / sqrt(var(x, dim=batch) + eps)
```

**Layer Normalization** normalizes across the feature dimension:

```python
# For each sample, compute mean and variance across features
x_norm = (x - mean(x, dim=features)) / sqrt(var(x, dim=features) + eps)
```

Which should you use? The answer depends on your architecture and task.

For sequences and proteins, **Layer Normalization is usually preferred**. Here is why. Batch normalization's statistics depend on the batch, which creates two problems. First, with variable-length protein sequences, batches often contain a mix of short and long proteins, making batch statistics noisy. Second, batch normalization behaves differently during training (uses batch statistics) and evaluation (uses running statistics), which can cause unexpected behavior.

Layer normalization avoids both issues because it normalizes each sequence independently:

```python
class ProteinBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Pre-norm architecture (more stable for deep networks)
        residual = x
        x = self.layernorm(x)
        x = self.linear1(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x
```

The placement of normalization matters. "Pre-norm" (normalize before the transformation) is more stable for very deep networks. "Post-norm" (normalize after the residual addition) was the original formulation but can be unstable without careful initialization.

### 2.5 Early Stopping: When to Call it Quits

Early stopping is a form of regularization based on time rather than architecture. The idea is simple: monitor validation performance during training and stop when it stops improving.

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # New best model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

# Usage in training loop
early_stopping = EarlyStopping(patience=15)
for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)

    if early_stopping.step(val_loss):
        torch.save(model.state_dict(), 'best_model.pt')

    if early_stopping.should_stop:
        print(f"Early stopping at epoch {epoch}")
        break

# Load best model for final evaluation
model.load_state_dict(torch.load('best_model.pt'))
```

The patience parameter controls how long to wait for improvement. For protein models, where training can be noisy, a patience of 10-20 epochs is common. Too little patience risks stopping during a temporary plateau; too much patience wastes compute and risks overfitting.

---

## 3. Learning Rate Schedules

### 3.1 Why Learning Rate is So Important

If there is one hyperparameter that matters more than all others, it is the learning rate. Too high, and your optimization diverges or oscillates wildly. Too low, and training takes forever and may get stuck in poor local minima.

The optimal learning rate is not constant throughout training. Early in training, we can take large steps because we are far from any minimum. Later, we need smaller steps to fine-tune and avoid overshooting. This is the motivation for learning rate schedules.

### 3.2 Learning Rate Warmup

Warmup is the practice of starting with a very small learning rate and gradually increasing it over the first few hundred or thousand steps. This sounds counterintuitive but is crucial for stable training of large models.

The reason relates to the statistics of gradients. At the start of training, the model's weights are essentially random, so gradients can be unusually large or point in misleading directions. Taking large steps based on these noisy gradients can push the model into bad regions of parameter space from which recovery is difficult.

Warmup is especially important when using adaptive optimizers like Adam. The optimizer needs to accumulate gradient statistics to properly scale learning rates for each parameter. Starting with small updates gives it time to build accurate statistics.

```python
def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Usage
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps=1000, total_steps=100000)

for step, batch in enumerate(train_loader):
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    scheduler.step()  # Update learning rate after each step
    optimizer.zero_grad()
```

A common rule of thumb: warmup for about 5-10% of total training steps. For protein models, this might mean 1000-5000 warmup steps.

### 3.3 Cosine Annealing

Cosine annealing is an elegant schedule that decreases the learning rate following a cosine curve:

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))$$

This gives a smooth decay that starts slow, accelerates in the middle, and slows down again at the end. The gradual approach to the minimum learning rate helps the optimizer settle into good solutions.

```python
# PyTorch's built-in cosine scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_epochs,  # Period of cosine
    eta_min=1e-6  # Minimum learning rate
)

# For step-wise updates (common in protein training)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=1e-6
)
```

A powerful combination is warmup followed by cosine decay:

```python
def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### 3.4 Reduce on Plateau

Sometimes we do not know in advance how many steps training will take. Reduce on plateau is an adaptive schedule that decreases the learning rate when progress stalls:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',  # 'min' for loss, 'max' for accuracy
    factor=0.5,  # Multiply LR by this when reducing
    patience=5,  # Wait this many epochs before reducing
    min_lr=1e-6
)

# In training loop
for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = evaluate(model, val_loader)
    scheduler.step(val_loss)  # Pass the monitored metric
```

This is particularly useful for exploratory experiments where you are not sure how long training should take.

---

## 4. Hyperparameter Tuning

### 4.1 What to Tune (and What to Leave Alone)

Not all hyperparameters are created equal. Here is a rough hierarchy of importance for protein deep learning:

**Critical (always tune):**
- Learning rate: Often the difference between success and failure
- Batch size: Affects both optimization dynamics and memory
- Model capacity (hidden dimensions, number of layers)

**Important (tune for best results):**
- Weight decay
- Dropout rate
- Learning rate schedule parameters (warmup steps, etc.)

**Usually fine with defaults:**
- Optimizer parameters (Adam's beta1, beta2)
- Layer normalization epsilon
- Specific architectural choices within a family

### 4.2 Grid Search and Random Search

The simplest tuning approach is grid search: define a grid of hyperparameter combinations and try them all.

```python
learning_rates = [1e-5, 1e-4, 1e-3]
batch_sizes = [16, 32, 64]
hidden_dims = [256, 512]

results = []
for lr in learning_rates:
    for bs in batch_sizes:
        for hidden in hidden_dims:
            val_loss = train_and_evaluate(lr=lr, batch_size=bs, hidden_dim=hidden)
            results.append({
                'lr': lr, 'batch_size': bs, 'hidden_dim': hidden,
                'val_loss': val_loss
            })
```

Grid search has a major weakness: it spends too many trials on unimportant hyperparameters. If learning rate matters a lot but batch size does not, a grid over both wastes most of its budget on batch size variations.

Random search is often better. By sampling hyperparameters randomly, we get more coverage of the important dimensions:

```python
import random

def sample_hyperparameters():
    return {
        'lr': 10 ** random.uniform(-5, -3),  # Log-uniform between 1e-5 and 1e-3
        'batch_size': random.choice([16, 32, 64, 128]),
        'hidden_dim': random.choice([256, 384, 512, 768]),
        'dropout': random.uniform(0.1, 0.4),
        'weight_decay': 10 ** random.uniform(-4, -1),
        'warmup_ratio': random.uniform(0.05, 0.15)
    }

# Run random search
n_trials = 50
results = []
for trial in range(n_trials):
    params = sample_hyperparameters()
    val_loss = train_and_evaluate(**params)
    results.append({'params': params, 'val_loss': val_loss})

best_trial = min(results, key=lambda x: x['val_loss'])
print(f"Best params: {best_trial['params']}")
```

### 4.3 Practical Tuning Strategy

Here is a practical approach that balances thoroughness with efficiency:

1. **Start with a fast sanity check.** Train for just a few epochs with default hyperparameters. Can the model overfit a tiny dataset? If not, something is fundamentally wrong.

2. **Find a reasonable learning rate.** Do a quick learning rate sweep, watching for the transition from stable learning to divergence.

3. **Tune the critical parameters together.** Learning rate and batch size interact, so tune them jointly.

4. **Lock in architecture, then tune regularization.** Once the basic architecture is working, adjust dropout and weight decay to control overfitting.

```python
def learning_rate_finder(model, train_loader, min_lr=1e-7, max_lr=1, num_steps=100):
    """Find the optimal learning rate by observing loss trajectory."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr)

    lr_schedule = np.geomspace(min_lr, max_lr, num_steps)
    losses = []
    lrs = []

    model.train()
    data_iter = iter(train_loader)

    for i, lr in enumerate(lr_schedule):
        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get batch (cycle if needed)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Training step
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        lrs.append(lr)

        # Stop if loss explodes
        if loss.item() > 4 * min(losses):
            break

    return lrs, losses
    # Plot and choose LR where loss is still decreasing steeply
```

---

## 5. Debugging Neural Networks

### 5.1 Common Failure Modes

When training goes wrong, it usually manifests in one of a few ways. Learning to recognize these patterns will save you hours of debugging.

**Loss not decreasing at all.** This often indicates a learning rate problem (too high or too low) or a bug in your code. Try a smaller learning rate first. If that does not help, verify your loss function and data loading.

**Loss decreasing then plateauing high.** The model has limited capacity or is stuck in a local minimum. Try increasing model size or using a different initialization.

**Training loss good, validation loss bad.** Classic overfitting. Increase regularization (dropout, weight decay) or get more data.

**NaN or Inf losses.** Numerical instability, often from too-high learning rate, missing gradient clipping, or problematic operations (log of zero, division by small numbers).

**Gradients vanishing.** Very deep networks without residual connections can have vanishing gradients. Check gradient norms during training.

### 5.2 Diagnostic Tools

**Loss curves** are your primary diagnostic tool. Always log both training and validation loss at regular intervals:

```python
from collections import defaultdict

class MetricsLogger:
    def __init__(self):
        self.metrics = defaultdict(list)

    def log(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)

    def plot(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        axes[0].plot(self.metrics['train_loss'], label='Train')
        axes[0].plot(self.metrics['val_loss'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].set_title('Loss Curves')

        # Learning rate
        axes[1].plot(self.metrics['lr'])
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')

        plt.tight_layout()
        return fig
```

**Gradient histograms** reveal whether gradients are flowing properly through your network:

```python
def log_gradient_stats(model):
    """Log gradient statistics for debugging."""
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            stats[name] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'max': grad.abs().max().item(),
                'norm': grad.norm().item()
            }
    return stats

# In training loop
if step % 100 == 0:
    grad_stats = log_gradient_stats(model)
    # Check for vanishing gradients (very small norms)
    # or exploding gradients (very large norms)
```

### 5.3 The Sanity Check Approach

Before training on your full dataset, run a series of sanity checks:

**Can you overfit a single batch?** If not, your model cannot learn at all.

```python
def sanity_check_overfit_single_batch(model, train_loader, num_steps=1000):
    """Verify model can memorize a single batch."""
    batch = next(iter(train_loader))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    initial_loss = None
    for step in range(num_steps):
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()

        if step % 100 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    final_loss = loss.item()
    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Ratio: {initial_loss / final_loss:.1f}x improvement")

    if final_loss > initial_loss * 0.1:
        print("WARNING: Model may not be learning properly!")
```

**Is your loss at initialization reasonable?** For classification with N classes, initial cross-entropy loss should be around log(N).

```python
def check_initial_loss(model, train_loader, num_classes):
    """Verify initial loss is reasonable."""
    model.eval()
    batch = next(iter(train_loader))

    with torch.no_grad():
        loss = compute_loss(model, batch)

    expected_loss = np.log(num_classes)
    print(f"Initial loss: {loss.item():.4f}")
    print(f"Expected for random: {expected_loss:.4f}")

    if abs(loss.item() - expected_loss) > expected_loss * 0.5:
        print("WARNING: Initial loss seems wrong. Check loss function or output layer.")
```

---

## 6. Protein-Specific Challenges

### 6.1 Handling Variable-Length Sequences

Proteins range from tens to thousands of residues. Naive approaches waste compute on padding or fail on long sequences. Here are practical solutions.

**Padding with masking** is the simplest approach. Pad shorter sequences in a batch to match the longest, then mask the loss computation:

```python
def collate_protein_batch(batch):
    """Custom collate function for variable-length proteins."""
    sequences = [item['sequence'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Find max length in batch
    max_len = max(seq.shape[0] for seq in sequences)

    # Pad sequences
    padded_seqs = torch.zeros(len(sequences), max_len, sequences[0].shape[-1])
    masks = torch.zeros(len(sequences), max_len)
    padded_labels = torch.zeros(len(sequences), max_len, dtype=torch.long)

    for i, (seq, label) in enumerate(zip(sequences, labels)):
        length = seq.shape[0]
        padded_seqs[i, :length] = seq
        masks[i, :length] = 1
        padded_labels[i, :length] = label

    return {
        'sequences': padded_seqs,
        'labels': padded_labels,
        'masks': masks
    }

def masked_cross_entropy(logits, labels, mask):
    """Compute loss only on non-padded positions."""
    # Flatten everything
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = labels.view(-1)
    mask_flat = mask.view(-1)

    # Compute per-element loss
    loss = nn.functional.cross_entropy(logits_flat, labels_flat, reduction='none')

    # Apply mask and reduce
    masked_loss = (loss * mask_flat).sum() / mask_flat.sum()
    return masked_loss
```

**Length bucketing** reduces padding waste by batching proteins of similar length together:

```python
from torch.utils.data import Sampler

class LengthBucketSampler(Sampler):
    def __init__(self, lengths, batch_size, bucket_size=100):
        self.batch_size = batch_size

        # Sort indices by length
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        # Create buckets
        self.batches = []
        for i in range(0, len(sorted_indices), batch_size):
            batch = sorted_indices[i:i + batch_size]
            self.batches.append(batch)

        # Shuffle batches (not within batches, to keep similar lengths together)
        np.random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
```

### 6.2 Sequence Identity Splits for Proper Evaluation

This is perhaps the most common mistake in protein machine learning, and it leads to wildly overoptimistic performance estimates.

Proteins in training and test sets must be sufficiently dissimilar in sequence. Why? Because proteins with similar sequences usually have similar structures and functions. If your test set contains proteins that are 90% identical to training proteins, you are not testing generalization, you are testing memorization.

The standard approach is to cluster proteins by sequence identity (commonly at 30% or 40%) and then split at the cluster level:

```python
import subprocess

def create_sequence_identity_splits(fasta_file, identity_threshold=0.3, train_ratio=0.8):
    """
    Create train/val/test splits that respect sequence identity.
    Requires MMseqs2 or CD-HIT to be installed.
    """
    # Run MMseqs2 clustering
    subprocess.run([
        'mmseqs', 'easy-cluster',
        fasta_file,
        'clusters',
        'tmp',
        '--min-seq-id', str(identity_threshold)
    ])

    # Parse cluster results
    clusters = parse_cluster_file('clusters_cluster.tsv')

    # Shuffle clusters
    cluster_ids = list(clusters.keys())
    np.random.shuffle(cluster_ids)

    # Split clusters (not proteins!)
    n_clusters = len(cluster_ids)
    n_train = int(n_clusters * train_ratio)
    n_val = int(n_clusters * 0.1)

    train_clusters = cluster_ids[:n_train]
    val_clusters = cluster_ids[n_train:n_train + n_val]
    test_clusters = cluster_ids[n_train + n_val:]

    # Get protein IDs for each split
    train_ids = [pid for c in train_clusters for pid in clusters[c]]
    val_ids = [pid for c in val_clusters for pid in clusters[c]]
    test_ids = [pid for c in test_clusters for pid in clusters[c]]

    return train_ids, val_ids, test_ids
```

Be especially careful when working with families or superfamilies. Even at 30% sequence identity, proteins from the same superfamily may share structural features that inflate test performance.

### 6.3 Dealing with Class Imbalance

Many protein prediction tasks have severe class imbalance. In contact prediction, only about 2-3% of residue pairs are in contact. In function prediction, rare functions might have only a handful of examples.

**Weighted loss functions** upweight rare classes:

```python
def compute_class_weights(labels, num_classes):
    """Compute inverse frequency weights."""
    counts = torch.bincount(labels.flatten(), minlength=num_classes).float()
    weights = 1.0 / (counts + 1)  # Add 1 to avoid division by zero
    weights = weights / weights.sum() * num_classes  # Normalize
    return weights

# Usage
class_weights = compute_class_weights(train_labels, num_classes=3)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Focal loss** focuses training on hard examples by downweighting easy ones:

```python
def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """
    Focal loss for imbalanced classification.
    gamma: focusing parameter (higher = more focus on hard examples)
    alpha: class balancing weight
    """
    ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # Probability of correct class
    focal_weight = (1 - pt) ** gamma
    loss = alpha * focal_weight * ce_loss
    return loss.mean()
```

For contact prediction specifically, the extreme imbalance often requires specialized approaches:

```python
def balanced_contact_loss(pred, target, positive_weight=10.0):
    """
    Loss function for contact prediction with severe imbalance.
    pred: (batch, L, L) logits
    target: (batch, L, L) binary labels
    """
    # Mask diagonal and short-range
    L = pred.size(-1)
    mask = torch.abs(
        torch.arange(L).unsqueeze(0) - torch.arange(L).unsqueeze(1)
    ) >= 6  # Only consider residues 6+ apart
    mask = mask.to(pred.device)

    # Flatten
    pred_flat = pred[mask]
    target_flat = target[mask]

    # Weighted BCE
    pos_weight = torch.tensor([positive_weight]).to(pred.device)
    loss = nn.functional.binary_cross_entropy_with_logits(
        pred_flat, target_flat.float(), pos_weight=pos_weight
    )
    return loss
```

### 6.4 Memory Management for Long Proteins

Proteins can be thousands of residues long, and attention-based models have O(L^2) memory requirements. Here are strategies for handling long sequences.

**Gradient checkpointing** trades compute for memory by recomputing activations during the backward pass instead of storing them:

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            # Checkpoint saves memory at cost of ~30% more compute
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

**Mixed precision training** uses float16 for most operations, roughly halving memory use:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()

    with autocast():  # Operations run in float16 where safe
        outputs = model(batch['sequences'])
        loss = criterion(outputs, batch['labels'])

    # Scale loss, backward, and update
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Dynamic batching** adjusts batch size based on sequence length:

```python
def create_dynamic_batches(dataset, max_tokens=4096):
    """Create batches that respect a maximum token budget."""
    lengths = [len(item['sequence']) for item in dataset]
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)

    batches = []
    current_batch = []
    current_tokens = 0
    max_len_in_batch = 0

    for idx in sorted_indices:
        seq_len = lengths[idx]

        # Check if adding this sequence would exceed budget
        new_max_len = max(max_len_in_batch, seq_len)
        new_tokens = new_max_len * (len(current_batch) + 1)

        if new_tokens > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = [idx]
            max_len_in_batch = seq_len
        else:
            current_batch.append(idx)
            max_len_in_batch = new_max_len

    if current_batch:
        batches.append(current_batch)

    return batches
```

---

## Key Takeaways

1. **Training is empirical science.** Success requires systematic experimentation and careful observation, not just clever architectures.

2. **Regularization prevents overfitting.** Use dropout, weight decay, and early stopping. Layer normalization is usually better than batch normalization for proteins.

3. **Learning rate is critical.** Always use warmup. Cosine annealing is a safe default schedule. Use learning rate finder for initial tuning.

4. **Debug systematically.** Start with sanity checks. Monitor loss curves and gradient statistics. Know the common failure modes.

5. **Protein data has unique challenges.** Use sequence identity splits to avoid data leakage. Handle variable lengths with masking or bucketing. Address class imbalance with weighted losses.

6. **Memory matters for long proteins.** Use gradient checkpointing, mixed precision, and dynamic batching.

---

## Exercises

1. **Learning rate finder.** Implement a learning rate finder and use it to find good learning rates for a secondary structure predictor. Plot the loss vs learning rate curve and identify the optimal range.

2. **Regularization ablation.** Train the same model with different combinations of regularization (none, dropout only, weight decay only, both). Compare train/validation curves. What do you observe?

3. **Data leakage experiment.** Train a model using random splits vs sequence identity splits at 30%. Compare test performance. By how much does using proper splits change the results?

4. **Memory optimization.** Take a transformer model that runs out of memory on 1000-residue proteins. Apply gradient checkpointing and mixed precision. How much longer can your sequences now be?

---

## References

- Srivastava et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
- Ioffe & Szegedy (2015). "Batch Normalization: Accelerating Deep Network Training"
- Ba et al. (2016). "Layer Normalization"
- Smith (2017). "Cyclical Learning Rates for Training Neural Networks"
- Loshchilov & Hutter (2017). "SGDR: Stochastic Gradient Descent with Warm Restarts"
- Zhang et al. (2019). "Which Algorithmic Choices Matter at Which Batch Sizes?"
- Rao et al. (2019). "Evaluating Protein Transfer Learning with TAPE" - Best practices for protein ML evaluation
- Rives et al. (2021). "Biological Structure and Function Emerge from Scaling Unsupervised Learning"
