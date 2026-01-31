# Lecture 3: Introduction to AI and Deep Learning

## Learning Objectives

By the end of this lecture, you will be able to:
1. Understand the machine learning pipeline from data to deployment
2. Work fluently with PyTorch tensors and autograd
3. Build and train neural networks from scratch
4. Apply deep learning to protein property prediction

## Prerequisites

- Lecture 1: Python and Data Science Basics
- Lecture 2: Protein Representations for Machine Learning
- Basic linear algebra (matrix multiplication, gradients)

---

## 1. What Does It Mean for a Computer to "Learn" About Proteins?

What does it mean for a computer to "learn" about proteins? This seemingly simple question opens the door to one of the most transformative developments in computational biology. When we say a machine learning model has "learned" to predict protein solubility, we mean something quite specific: the model has discovered numerical patterns in the data that allow it to make accurate predictions on proteins it has never seen before.

Consider the challenge facing a biochemist trying to express a protein in E. coli. Some proteins dissolve beautifully in aqueous solution, ready for crystallography or functional studies. Others aggregate into insoluble inclusion bodies, requiring laborious refolding protocols that often fail. For decades, predicting which proteins would be soluble required expensive experimental screening or reliance on crude heuristics. Today, machine learning models can make these predictions in milliseconds with remarkable accuracy.

But how does this actually work? The answer lies in understanding that proteins, despite their immense complexity, exhibit statistical regularities. Soluble proteins tend to have certain amino acid compositions, charge distributions, and hydrophobicity patterns. A machine learning model does not understand biology in the way a scientist does. Instead, it discovers these patterns automatically from examples, encoding them as numerical weights that transform an input protein sequence into a prediction.

### The Machine Learning Pipeline

The journey from raw biological data to a deployed prediction model follows a structured pipeline. Each stage presents unique challenges when working with proteins.

**Data Collection** begins with gathering proteins and their associated labels. For solubility prediction, this might mean mining databases like UniProt for experimentally validated soluble proteins, or analyzing high-throughput expression studies. The quality of your data fundamentally limits what any model can learn. Garbage in, garbage out applies forcefully here.

**Preprocessing** transforms raw data into a clean, consistent format. Protein sequences may contain ambiguous amino acids (B for Asp or Asn, X for unknown), unusual characters, or metadata that needs parsing. Structure data from the PDB requires validation for missing atoms, alternate conformations, and resolution quality.

**Feature Engineering** is where domain knowledge meets machine learning. As we explored in Lecture 2, proteins can be represented as one-hot encodings, physicochemical feature vectors, learned embeddings, or graphs. The choice of representation profoundly affects what patterns a model can discover.

**Model Training** is where learning actually happens. The model sees thousands of proteins with known labels, gradually adjusting its internal parameters to minimize prediction errors. This stage involves critical choices about model architecture, optimization algorithms, and regularization strategies.

**Evaluation** measures how well the model generalizes to new proteins. This is trickier than it sounds for proteins because related sequences often have similar properties. A naive train/test split might allow the model to "cheat" by memorizing similar proteins. Proper evaluation requires sequence-identity-aware splitting to ensure the test set contains truly novel proteins.

**Deployment** brings the model into production where it can make predictions on new proteins. This stage introduces practical constraints around inference speed, memory usage, and integration with existing workflows.

### Flavors of Machine Learning

Machine learning comes in several distinct flavors, each suited to different problems.

**Supervised learning** is the most straightforward: you have input-output pairs and want to learn the mapping between them. Given protein sequences and their solubility labels, learn to predict solubility. Given structures and stability measurements, learn to predict stability. The key requirement is labeled data, which in biology often comes from expensive experiments.

**Unsupervised learning** works without labels, instead discovering structure in the data itself. Clustering similar protein sequences, learning low-dimensional embeddings that capture evolutionary relationships, or identifying protein families all fall into this category. The power of unsupervised learning is that it can leverage the vast quantities of unlabeled sequence data available in databases like UniProt.

**Self-supervised learning** has revolutionized protein machine learning in recent years. The key insight is to create supervision signals from the data itself. Mask out 15% of amino acids in a protein sequence and train the model to predict them back. This simple task, borrowed from natural language processing, forces the model to learn deep representations of protein sequence-structure-function relationships. Models like ESM and ProtTrans, trained on hundreds of millions of protein sequences with this approach, have become foundational tools in computational biology.

### Matching Problems to Formulations

Different biological questions map to different machine learning formulations. Understanding this mapping is crucial for successful applications.

**Regression** problems have continuous outputs. Predicting a protein's melting temperature (Tm) or binding affinity (Kd) are regression tasks. The model outputs a number, and we measure error as the difference between prediction and truth.

**Binary classification** distinguishes two categories. Is this protein an enzyme or not? Will it localize to the membrane? The model outputs a probability, and we threshold it to make a decision.

**Multi-class classification** extends this to more categories. Predicting which of 20 secondary structure states each residue adopts, or classifying proteins into major functional categories.

**Multi-label classification** handles cases where multiple labels can apply simultaneously. A protein might be both an enzyme AND membrane-bound. Each label is predicted independently.

**Sequence-to-sequence** tasks predict an output for each position in the input. Secondary structure prediction produces one of three states (helix, sheet, coil) for every residue. Disorder prediction identifies which residues lack fixed structure.

---

## 2. PyTorch: Your Laboratory for Neural Networks

If machine learning is the science, PyTorch is the laboratory equipment. Just as a biochemist needs pipettes, centrifuges, and spectrophotometers, a computational biologist needs tools for constructing and training neural networks. PyTorch, developed by Meta AI Research, has become the dominant framework for deep learning research, including the models that have transformed protein science.

Why PyTorch specifically? Several reasons. First, its "eager execution" model means code runs line by line, making debugging intuitive. Second, its design closely mirrors how researchers think about computation, making translation from mathematical ideas to working code straightforward. Third, the entire ecosystem of protein machine learning, from ESM to AlphaFold's OpenFold implementation, builds on PyTorch.

### Tensors: The Atoms of Deep Learning

At the heart of PyTorch lies the tensor. If you know NumPy arrays, tensors will feel familiar. They are multi-dimensional arrays of numbers. A single number is a 0-dimensional tensor (a scalar). A list of numbers is a 1-dimensional tensor (a vector). A table of numbers is a 2-dimensional tensor (a matrix). Higher dimensions are also possible: a batch of protein sequences might be represented as a 3-dimensional tensor with dimensions (batch_size, sequence_length, features).

```python
import torch

# Creating tensors
x = torch.zeros(3, 4)           # 3x4 matrix of zeros
x = torch.ones(3, 4)            # 3x4 matrix of ones
x = torch.randn(3, 4)           # Random values from standard normal distribution
x = torch.tensor([1, 2, 3])     # From a Python list
x = torch.from_numpy(np_array)  # From a NumPy array

# Inspecting tensor properties
print(x.shape)     # torch.Size([3, 4]) - the dimensions
print(x.dtype)     # torch.float32 - the data type
print(x.device)    # cpu or cuda:0 - where the tensor lives
```

What makes tensors special compared to NumPy arrays? Two things: GPU acceleration and automatic differentiation. Let us explore each.

### The GPU Advantage

Modern GPUs contain thousands of simple processors that can perform arithmetic operations in parallel. A single NVIDIA GPU can execute trillions of floating-point operations per second. This massive parallelism is perfect for the matrix multiplications at the heart of neural networks.

Moving computation to the GPU is straightforward in PyTorch:

```python
# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move a tensor to the GPU
x = torch.randn(1000, 1000)
x_gpu = x.to(device)

# Now all operations happen on the GPU
y_gpu = x_gpu @ x_gpu.T  # Matrix multiplication, computed on GPU
```

For the matrix multiplication above, a GPU might be 50-100x faster than a CPU. When training neural networks that involve millions of such operations, this speedup is the difference between experiments taking hours versus weeks.

### Tensor Operations

Tensors support all the arithmetic operations you would expect, with the same broadcasting rules as NumPy:

```python
a = torch.randn(3, 4)
b = torch.randn(3, 4)

# Element-wise operations
c = a + b          # Addition
c = a * b          # Multiplication (element-wise!)
c = a ** 2         # Squaring each element

# Matrix multiplication (different from element-wise!)
c = a @ b.T        # 3x4 multiplied by 4x3 gives 3x3
c = torch.matmul(a, b.T)  # Equivalent

# Broadcasting: smaller tensor is "stretched" to match
a = torch.randn(3, 4)
b = torch.randn(4)     # Just 4 elements
c = a + b              # b is broadcast across all 3 rows

# Reductions summarize tensors
x.sum()            # Sum of all elements
x.sum(dim=0)       # Sum along first dimension
x.mean(dim=-1)     # Mean along last dimension
x.max(dim=1)       # Maximum along second dimension
```

### Reshaping: The Art of Tensor Origami

Neural networks constantly reshape data as it flows through layers. A batch of protein sequences might start as (batch, length, 20) for one-hot encodings, then become (batch, 128, length) after an embedding layer with 128 features. Mastering tensor reshaping is essential.

```python
x = torch.randn(2, 3, 4)

# Reshape: change dimensions while preserving total elements
x.view(6, 4)         # Flatten first two dimensions
x.reshape(2, 12)     # Same idea, more flexible

# Transpose and permute: reorder dimensions
x.transpose(0, 1)    # Swap dimensions 0 and 1: (3, 2, 4)
x.permute(2, 0, 1)   # Arbitrary reordering: (4, 2, 3)

# Add/remove singleton dimensions
x.unsqueeze(0)       # Add dimension at position 0: (1, 2, 3, 4)
x.squeeze()          # Remove all size-1 dimensions
```

---

## 3. Automatic Differentiation: Teaching Computers Calculus

Here is a profound question: how does a neural network learn? The answer involves calculus, but not the kind you might remember from undergraduate courses involving tedious symbol manipulation. Instead, neural networks learn through a beautiful algorithmic trick called automatic differentiation.

### The Intuition Behind Learning

Imagine you are trying to predict protein solubility. Your model takes a protein sequence and outputs a number between 0 and 1 representing predicted probability of solubility. For a protein you know to be soluble (label = 1), if your model predicts 0.3, it is wrong. We quantify this wrongness with a loss function, perhaps something like $(1 - 0.3)^2 = 0.49$.

Now comes the key insight: your model's prediction depends on its internal parameters (weights). Different weight values would give different predictions. Some weight values would make the prediction closer to 1 (reducing the loss), while others would make it worse. Learning means finding weight values that minimize the loss across all training proteins.

How do we find these good weight values? We use gradients. The gradient of the loss with respect to a weight tells us: "if I increase this weight slightly, how does the loss change?" If increasing a weight would increase the loss, we should decrease it. If increasing it would decrease the loss, we should increase it. This is gradient descent.

### The Chain Rule: Propagating Blame

Neural networks compose many simple functions. The output of one layer feeds into the next, which feeds into the next, and so on. To compute how a weight in an early layer affects the final loss, we need the chain rule from calculus:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$$

In words: to find how $x$ affects $L$, multiply how $y$ affects $L$ by how $x$ affects $y$. This simple rule, applied recursively backward through the network, gives us gradients for all weights. This is the "backpropagation" algorithm, and it is what makes neural networks trainable.

### PyTorch's Autograd Magic

The remarkable thing about PyTorch is that you never have to implement backpropagation yourself. You simply define the forward computation, and PyTorch automatically builds a computational graph that tracks all operations. When you call `.backward()`, it traverses this graph in reverse, computing all gradients.

```python
# Enable gradient tracking with requires_grad=True
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Forward computation
y = x ** 2 + 3 * x
z = y.sum()  # Loss must be a scalar for .backward()

# Backward pass computes gradients
z.backward()

# Gradients are stored in the .grad attribute
print(x.grad)  # tensor([7., 9.])
```

Let us verify this manually. We have $z = \sum_i (x_i^2 + 3x_i)$, so $\frac{\partial z}{\partial x_i} = 2x_i + 3$. For $x_1 = 2$: $2(2) + 3 = 7$. For $x_2 = 3$: $2(3) + 3 = 9$. PyTorch computed this automatically.

### Practical Autograd

Here is a more realistic example: linear regression with learnable weights.

```python
# Data: predicting some property from features
X = torch.randn(100, 10)  # 100 proteins, 10 features each
y = torch.randn(100, 1)   # True property values

# Learnable parameters (requires_grad=True is crucial!)
W = torch.randn(10, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Forward pass
y_pred = X @ W + b
loss = ((y_pred - y) ** 2).mean()  # Mean squared error

# Backward pass
loss.backward()

# Now W.grad contains dL/dW, b.grad contains dL/db
print(f"Weight gradient shape: {W.grad.shape}")  # (10, 1)
```

The gradient `W.grad` tells us how to adjust each weight to reduce the loss. This is the foundation of training neural networks.

### Turning Off Gradient Tracking

Computing gradients uses memory (to store the computational graph) and computation. During inference, when you just want predictions without training, you should disable gradient tracking:

```python
# Context manager approach
with torch.no_grad():
    y_pred = model(x)
    # No computational graph built, more memory efficient

# Decorator approach for functions
@torch.no_grad()
def predict(model, x):
    return model(x)
```

---

## 4. Building Neural Networks: From Neurons to Architectures

With tensors and autograd understood, we can now build neural networks. But what exactly is a neural network? Let us build up from first principles, using protein examples throughout.

### The Single Neuron

The fundamental unit is the artificial neuron. It takes multiple inputs, computes a weighted sum, adds a bias, and applies a non-linear function:

$$\text{output} = \sigma(w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b)$$

where $\sigma$ is an activation function like ReLU (max(0, x)) or sigmoid ($\frac{1}{1+e^{-x}}$).

For proteins, the inputs might be features like hydrophobicity, charge, and length. The output might represent predicted solubility. The weights determine how much each feature contributes. The bias shifts the decision boundary. The activation introduces non-linearity, allowing the neuron to model complex relationships.

### Layers: Many Neurons in Parallel

A single neuron is limited. But arrange many neurons in parallel, each receiving the same inputs but with different weights, and you get a layer. If we have 64 neurons, we get 64 different weighted combinations of the input features. This can be written compactly as matrix multiplication:

$$\mathbf{h} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$$

where $\mathbf{W}$ is a matrix of weights, $\mathbf{x}$ is the input vector, and $\mathbf{h}$ is the output vector.

### Depth: Stacking Layers

The real power of neural networks comes from stacking multiple layers. The output of one layer becomes the input to the next. Each layer can learn increasingly abstract representations of the input.

For a protein classifier:
- Layer 1 might learn to detect individual amino acid properties
- Layer 2 might learn to recognize local motifs (charge clusters, hydrophobic patches)
- Layer 3 might learn to identify structural elements
- The output layer combines these into a prediction

This hierarchical learning is what makes deep learning "deep."

### nn.Module: PyTorch's Building Block

In PyTorch, all neural network components inherit from `nn.Module`. This base class provides machinery for tracking parameters, moving to GPU, saving/loading, and more.

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Define layers as attributes
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Define the forward computation
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create the model
model = SimpleNet(input_dim=20, hidden_dim=64, output_dim=2)

# Use it: just call like a function!
x = torch.randn(32, 20)  # Batch of 32 proteins, 20 features each
output = model(x)        # Shape: (32, 2)
```

The `__init__` method defines what layers exist. The `forward` method defines how data flows through them. PyTorch handles the backward pass automatically.

### Common Layers

PyTorch provides pre-built layers for common operations:

```python
# Linear layer: y = Wx + b
nn.Linear(in_features=20, out_features=64)

# Activations introduce non-linearity
nn.ReLU()      # max(0, x) - simple, effective, widely used
nn.GELU()      # Smooth approximation, used in transformers
nn.Sigmoid()   # Squashes to (0, 1), good for probabilities
nn.Softmax(dim=-1)  # Normalizes to probability distribution

# Normalization stabilizes training
nn.LayerNorm(normalized_shape=64)
nn.BatchNorm1d(num_features=64)

# Dropout prevents overfitting by randomly zeroing neurons
nn.Dropout(p=0.1)  # 10% dropout probability

# Embedding maps discrete tokens to continuous vectors
nn.Embedding(num_embeddings=21, embedding_dim=64)  # 21 amino acids
```

### Sequential: Quick Model Definition

For simple architectures where data flows straight through, `nn.Sequential` offers a shortcut:

```python
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 2)
)
```

This is equivalent to the class definition above but more concise.

### Managing Parameters

Neural networks can have millions of parameters. PyTorch provides tools to inspect and manage them:

```python
# List all parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# Count parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total:,}, Trainable: {trainable:,}")

# Save a trained model
torch.save(model.state_dict(), 'model.pt')

# Load it back
model.load_state_dict(torch.load('model.pt'))
```

---

## 5. Training Neural Networks: The Learning Story

Training a neural network is like teaching someone to recognize proteins. You show them examples, point out their mistakes, and let them adjust their understanding. Let us walk through this process step by step.

### Loss Functions: Measuring Mistakes

Before we can learn, we need a way to measure how wrong our predictions are. This is the job of the loss function. Different problems call for different loss functions.

**Mean Squared Error (MSE)** measures average squared difference between predictions and targets. It is the workhorse of regression:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Use MSE when predicting continuous values like binding affinity or stability. Squaring penalizes large errors heavily, which may or may not be what you want.

**Binary Cross-Entropy (BCE)** is designed for binary classification. It measures how well predicted probabilities match binary labels:

$$\text{BCE} = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

The intuition: if the true label is 1, we want $\hat{y}$ close to 1, making $\log(\hat{y})$ close to 0. If the true label is 0, we want $\hat{y}$ close to 0, making $\log(1-\hat{y})$ close to 0.

**Cross-Entropy** generalizes BCE to multiple classes:

$$\text{CE} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

In PyTorch:

```python
# Regression
criterion = nn.MSELoss()

# Binary classification (use with logits, includes sigmoid)
criterion = nn.BCEWithLogitsLoss()

# Multi-class classification (use with logits, includes softmax)
criterion = nn.CrossEntropyLoss()
```

### Optimizers: The Learning Strategy

Now we know how wrong our predictions are. But how do we improve? Optimizers take the gradients computed by backpropagation and use them to update the weights.

**Stochastic Gradient Descent (SGD)** is the simplest approach. Take a step in the direction that reduces the loss:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L$$

where $\eta$ is the learning rate controlling step size. Too small and learning is slow. Too large and training becomes unstable.

**Adam** (Adaptive Moment Estimation) is the most popular optimizer in practice. It maintains moving averages of both the gradient and its square, adapting the learning rate for each parameter:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}$$

Adam works well out of the box for most problems and is generally the default choice.

```python
# SGD with momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (most common choice)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# AdamW (Adam with proper weight decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

### The Training Loop: A Story in Four Acts

Training unfolds as a repeated cycle with four essential steps:

**Act 1: Forward Pass** - We feed a batch of proteins through the model, producing predictions. Data flows forward through the network, layer by layer.

**Act 2: Compute Loss** - We compare predictions to true labels using our loss function. This gives us a single number measuring how wrong we are.

**Act 3: Backward Pass** - We call `loss.backward()`, which computes gradients for all parameters. Mathematically, we are asking: "how should each weight change to reduce this loss?"

**Act 4: Update Weights** - The optimizer takes the gradients and updates the weights according to its strategy. We have learned from this batch.

```python
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # Enable training mode (affects dropout, batchnorm)
    total_loss = 0

    for batch_x, batch_y in dataloader:
        # Move data to device
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Act 1: Forward pass
        outputs = model(batch_x)

        # Act 2: Compute loss
        loss = criterion(outputs, batch_y)

        # Act 3: Backward pass
        optimizer.zero_grad()  # Clear old gradients first!
        loss.backward()

        # Optional: prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Act 4: Update weights
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
```

Note the critical `optimizer.zero_grad()` call. Gradients in PyTorch accumulate by default. Without zeroing, gradients from previous batches would contaminate the current update.

### Validation: Checking Our Work

Training loss alone can be misleading. The model might memorize training examples without learning generalizable patterns. We need a separate validation set to monitor true performance:

```python
@torch.no_grad()  # No gradients needed for evaluation
def evaluate(model, dataloader, criterion, device):
    model.eval()  # Disable dropout, use running stats for batchnorm
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        total_loss += loss.item()
        all_preds.append(outputs.cpu())
        all_labels.append(batch_y.cpu())

    return total_loss / len(dataloader), torch.cat(all_preds), torch.cat(all_labels)
```

### Putting It All Together

A complete training script orchestrates these pieces, adding learning rate scheduling and early stopping:

```python
def train_model(model, train_loader, val_loader, epochs=100, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Reduce learning rate when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validation phase
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        # Adjust learning rate
        scheduler.step(val_loss)

        # Early stopping: stop if validation loss stops improving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model
```

---

## 6. Data Loading: Feeding Proteins to Neural Networks

Neural networks are hungry. Training requires showing them thousands or millions of examples. Efficient data loading becomes crucial, especially when proteins have variable lengths and complex representations.

### The Dataset Class

PyTorch's `Dataset` class defines how to access individual examples. You implement two methods: `__len__` returns the total number of examples, and `__getitem__` returns one example by index.

```python
from torch.utils.data import Dataset, DataLoader

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, max_len=512):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
        # Map amino acids to integers
        self.aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]

        # Convert sequence to integer encoding
        encoded = torch.zeros(self.max_len, dtype=torch.long)
        for i, aa in enumerate(seq[:self.max_len]):
            if aa in self.aa_to_idx:
                encoded[i] = self.aa_to_idx[aa] + 1  # 0 reserved for padding

        # Track which positions are real vs padding
        mask = torch.zeros(self.max_len, dtype=torch.float)
        mask[:min(len(seq), self.max_len)] = 1.0

        return {
            'sequence': encoded,
            'mask': mask,
            'label': torch.tensor(label, dtype=torch.long)
        }
```

### The DataLoader: Batching and Shuffling

The `DataLoader` wraps a dataset, handling batching, shuffling, and parallel loading:

```python
train_dataset = ProteinDataset(train_seqs, train_labels)
val_dataset = ProteinDataset(val_seqs, val_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,       # Process 32 proteins at once
    shuffle=True,        # Randomize order each epoch
    num_workers=4,       # Parallel data loading threads
    pin_memory=True      # Faster GPU transfer
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,       # Can use larger batches for evaluation
    shuffle=False        # Keep consistent order for evaluation
)

# Iterate through batches
for batch in train_loader:
    sequences = batch['sequence']  # Shape: (batch_size, max_len)
    masks = batch['mask']          # Shape: (batch_size, max_len)
    labels = batch['label']        # Shape: (batch_size,)
```

### Handling Variable-Length Sequences

Proteins have different lengths. Padding everything to the longest possible sequence wastes computation. A custom collate function can pad each batch to its own maximum length:

```python
def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    max_len = max(len(item['sequence']) for item in batch)

    sequences = torch.zeros(len(batch), max_len, dtype=torch.long)
    masks = torch.zeros(len(batch), max_len)
    labels = torch.zeros(len(batch), dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = len(item['sequence'])
        sequences[i, :seq_len] = item['sequence']
        masks[i, :seq_len] = 1.0
        labels[i] = item['label']

    return {'sequence': sequences, 'mask': masks, 'label': labels}

loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

---

## 7. Case Study: Predicting Protein Solubility

Let us bring everything together with a real application: predicting whether a protein will be soluble when expressed in E. coli. This is a classic binary classification problem with immediate practical value.

### The Problem

Solubility prediction matters because expressing recombinant proteins is a core technique in structural biology, biotechnology, and therapeutic development. When a protein aggregates into inclusion bodies instead of dissolving in the cytoplasm, downstream applications become much harder. A model that predicts solubility can guide construct design and expression conditions.

What makes this problem amenable to machine learning? Solubility is influenced by sequence-level properties: amino acid composition, charge distribution, hydrophobicity patterns, and the presence of certain motifs. These patterns are learnable from data.

### The Model Architecture

For this task, we will build a convolutional neural network that processes amino acid embeddings:

```python
import torch.nn.functional as F

class ProteinClassifier(nn.Module):
    def __init__(self, vocab_size=21, embed_dim=64, hidden_dim=128, num_classes=2):
        super().__init__()

        # Learn continuous representations for amino acids
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Convolutional layers detect local patterns (motifs)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)

        # Final classification layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, mask=None):
        # x shape: (batch, seq_len) - integer-encoded amino acids

        # Embed: (batch, seq_len, embed_dim)
        x = self.embedding(x)

        # Conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)

        # Apply convolutions with ReLU activation
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))

        # Global average pooling over sequence length
        # Respects the mask to ignore padding positions
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, seq_len)
            x = (x * mask).sum(dim=2) / mask.sum(dim=2)
        else:
            x = x.mean(dim=2)

        # Classify
        x = self.fc(x)
        return x
```

The architecture reflects domain knowledge. Convolutional layers with kernel size 5 can detect patterns spanning 5 amino acids, which is relevant for local motifs. Global average pooling aggregates information across the entire sequence, appropriate for a sequence-level prediction.

### Training the Model

```python
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
df = pd.read_csv('solubility_data.csv')
print(f"Dataset: {len(df)} proteins")
print(f"Class balance:\n{df['label'].value_counts()}")

# Split data stratifying by label to maintain class balance
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'])

# Create datasets and loaders
train_dataset = ProteinDataset(train_df['sequence'].tolist(), train_df['label'].tolist())
val_dataset = ProteinDataset(val_df['sequence'].tolist(), val_df['label'].tolist())
test_dataset = ProteinDataset(test_df['sequence'].tolist(), test_df['label'].tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# Create and train model
model = ProteinClassifier()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

trained_model = train_model(model, train_loader, val_loader, epochs=50)
```

### Evaluation

Beyond simple accuracy, we examine multiple metrics to understand model performance:

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def evaluate_classifier(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch['sequence'].to(device)
            mask = batch['mask'].to(device)
            y = batch['label']

            logits = model(x, mask)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    auc = roc_auc_score(all_labels, all_probs)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")

    return accuracy, precision, recall, f1, auc
```

**Accuracy** tells us what fraction of predictions are correct. But for imbalanced datasets (many more soluble than insoluble proteins, or vice versa), accuracy can be misleading.

**Precision** measures: of the proteins we predicted as soluble, how many actually were? High precision means few false positives.

**Recall** measures: of the proteins that were actually soluble, how many did we catch? High recall means few false negatives.

**F1 Score** balances precision and recall, useful when you care about both.

**AUC-ROC** measures how well the model separates classes across all possible thresholds, providing a threshold-independent assessment.

---

## 8. Best Practices for Deep Learning

Training neural networks is part science, part art. Here are battle-tested practices that will save you hours of debugging.

### Debugging Neural Networks

Neural networks fail silently. The code runs, loss decreases, but predictions are garbage. Systematic debugging is essential.

```python
# Check for NaN gradients (sign of numerical instability)
for name, param in model.named_parameters():
    if param.grad is not None and torch.isnan(param.grad).any():
        print(f"NaN gradient in {name}")

# Verify output ranges make sense
print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")

# Check that shapes match expectations
print(f"Input: {x.shape}, Output: {model(x).shape}")

# Use hooks to peek inside the network
def print_hook(module, input, output):
    print(f"{module.__class__.__name__}: {input[0].shape} -> {output.shape}")

model.fc1.register_forward_hook(print_hook)
```

### Reproducibility

Science requires reproducibility. Set all random seeds:

```python
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism (may hurt performance):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

Note that full determinism can slow training. For initial experiments, setting the Python, NumPy, and PyTorch seeds is usually sufficient.

### Mixed Precision Training

Modern GPUs have specialized hardware for 16-bit floating point (FP16) operations that is 2x faster than FP32. Mixed precision training uses FP16 where possible, FP32 where necessary:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch_x, batch_y in train_loader:
    optimizer.zero_grad()

    # Forward pass in FP16
    with autocast():
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

    # Backward pass handles scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

This typically gives 1.5-2x speedup with minimal accuracy impact.

---

## Key Takeaways

1. **Machine learning discovers patterns** in protein data automatically, encoding them as numerical weights that transform sequences into predictions.

2. **Tensors** are the fundamental data structure, combining NumPy-like operations with GPU acceleration and automatic differentiation.

3. **Autograd** implements the chain rule algorithmically, computing gradients for all parameters automatically from the forward computation.

4. **Neural networks** are compositions of simple layers. The `nn.Module` class provides the scaffolding for building them.

5. **Training** is a loop: forward pass, compute loss, backward pass, update weights. Early stopping and learning rate scheduling are essential for good results.

6. **Data loading** with Dataset and DataLoader handles batching, shuffling, and parallel processing. Custom collate functions manage variable-length sequences.

7. **Evaluation** must use held-out data and appropriate metrics. For proteins, sequence-identity-aware splits prevent data leakage.

---

## Exercises

1. Implement a 3-layer MLP for per-residue secondary structure prediction. Your model should output three values (helix, sheet, coil probabilities) for each amino acid position.

2. Add learning rate warmup to the training loop: start with a very small learning rate and linearly increase it over the first 1000 steps.

3. Implement gradient accumulation to train with larger effective batch sizes when GPU memory is limited.

4. Compare SGD (with momentum), Adam, and AdamW on the protein solubility task. Plot learning curves and final accuracy for each.

---

## References

- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). "Deep Learning" - Chapter 8: Optimization
- Kingma, D. P., & Ba, J. (2015). "Adam: A Method for Stochastic Optimization"
