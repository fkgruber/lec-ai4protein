# Lecture 9: RFDiffusion - Teaching Computers to Dream Up New Proteins

## Learning Objectives

By the end of this lecture, you will be able to:
1. Understand SE(3) equivariance and why it matters for protein generation
2. Explain diffusion on rigid body frames and SO(3) noise
3. Implement key components of RFDiffusion's architecture
4. Design conditional generation strategies for protein design

---

## 1. The Dream of De Novo Protein Design

What if we could design proteins from scratch?

For decades, this question has captivated biochemists, structural biologists, and bioengineers alike. Nature has spent billions of years evolving proteins to perform an astonishing variety of functions: catalyzing reactions, transporting molecules, providing structural support, and recognizing specific molecular targets with exquisite precision. But nature's evolutionary process is slow, meandering, and constrained by the need to maintain function at every step along the way. What if we could bypass evolution entirely and design proteins that nature never imagined?

This is the promise of **de novo protein design** - creating entirely new proteins that don't exist in nature, proteins engineered from first principles to perform specific functions. It's the difference between breeding horses for thousands of years versus designing a car from scratch. And in July 2023, a team at the University of Washington led by David Baker unveiled a tool that brought this dream tantalizingly close to reality: **RFDiffusion**.

RFDiffusion represents a paradigm shift in how we think about protein design. Unlike traditional approaches that start with existing protein templates and modify them, RFDiffusion generates novel protein backbones by learning to reverse a noise process - the same fundamental idea behind image generation models like DALL-E and Stable Diffusion, but adapted for the unique geometric constraints of molecular structures. The results have been nothing short of remarkable: novel proteins that fold as designed, bind to their targets, and even catalyze chemical reactions.

But how does one teach a computer to "dream up" a protein? How do you apply the mathematics of diffusion to objects that exist in three-dimensional space and can be rotated and translated without changing their fundamental nature? These questions lead us into some beautiful mathematics at the intersection of machine learning, group theory, and structural biology.

---

## 2. Why Geometry Matters: The Case for SE(3) Equivariance

### A Protein is a Protein, No Matter How You Turn It

Let's start with a seemingly obvious observation that turns out to have profound implications for how we build protein generative models.

Imagine you're looking at a protein structure on your computer screen. You grab it with your mouse and rotate it around, viewing it from different angles. From one direction, you might see a beautiful alpha-helix spiraling through space; from another angle, the same helix might appear as a circle of atoms. But here's the critical insight: **it's the same protein either way**.

The biological function of a protein - its ability to catalyze a reaction, bind to a partner, or form a structural element - depends on its shape, not on how that shape happens to be oriented in the laboratory coordinate system. An enzyme doesn't care whether it's pointing north or south. A receptor doesn't function differently because someone rotated the petri dish.

This property is called **SE(3) invariance**, and it's one of the most important symmetries in molecular biology. The name comes from mathematics: SE(3) stands for the "Special Euclidean group in three dimensions," which is a fancy way of describing all possible combinations of rotations and translations in 3D space. When we say a property is SE(3) invariant, we mean it doesn't change when you apply any rotation or translation to the molecule.

But there's a subtler requirement for a generative model. We don't just want the model's outputs to have the same properties regardless of orientation - we want the model itself to behave consistently under rotations and translations. If you rotate the input, the output should rotate in exactly the same way. This stronger requirement is called **SE(3) equivariance**.

Mathematically, a function $f$ is SE(3) equivariant if:

$$f(T \cdot x) = T \cdot f(x)$$

where $T \in SE(3)$ is any rotation and translation, and $T \cdot x$ means applying that transformation to $x$.

### Why Equivariance is Worth the Trouble

You might wonder: why go through all the mathematical complexity of building an equivariant model? Couldn't we just augment our training data with random rotations and hope the model learns to handle them?

The answer is yes, we could do that - and for many years, that was exactly the approach taken. But equivariant architectures offer several compelling advantages:

**Data efficiency.** An equivariant model knows, by construction, that all orientations of a protein are equivalent. It doesn't need to "learn" this from data. A non-equivariant model, by contrast, might need to see the same protein structure from many different angles before it generalizes this understanding. In a field where high-quality structural data is precious and limited, this efficiency matters.

**Physically meaningful representations.** When a model is equivariant, its internal representations have well-defined transformation properties. Vectors in the representation space transform like actual 3D vectors when you rotate the input. This makes the model's "thought process" more interpretable and more likely to capture physically meaningful relationships.

**Generalization.** Perhaps most importantly, equivariant models generalize better to unseen structures. They can't learn spurious correlations based on orientation - there are no such correlations to learn. This is particularly valuable for de novo design, where we want to generate proteins that may look quite different from anything in the training set.

---

## 3. The Language of Rotations

### Many Ways to Say the Same Thing

Before we can understand how to add noise to rotations (a key component of RFDiffusion), we need to understand how to represent rotations in the first place. This turns out to be surprisingly subtle, and the choice of representation has significant practical implications.

There are four common ways to represent a rotation in 3D:

**Rotation matrices.** A 3x3 orthogonal matrix $R$ with determinant +1 represents a rotation. This is the most direct representation: to rotate a vector $\vec{v}$, you simply compute $R\vec{v}$. The downside is over-parameterization - you need 9 numbers to describe something that has only 3 degrees of freedom (think of the three angles you need to orient a rigid body).

**Euler angles.** Three angles (often called roll, pitch, and yaw) that describe successive rotations about coordinate axes. This is intuitive and minimal, but suffers from a notorious problem called "gimbal lock" where certain orientations cause a loss of a degree of freedom.

**Axis-angle representation.** A rotation can be described by the axis around which you're rotating (a unit vector, 2 degrees of freedom) and the angle of rotation (1 degree of freedom). This is minimal and intuitive, but becomes singular for very small rotations where the axis is ill-defined.

**Quaternions.** Four numbers $(w, x, y, z)$ subject to the constraint $w^2 + x^2 + y^2 + z^2 = 1$. This might seem like overkill for a 3-DOF quantity, but quaternions have remarkable numerical properties: they're singularity-free, interpolate smoothly, and avoid many of the numerical issues that plague other representations.

RFDiffusion uses quaternions internally, precisely because of their numerical stability. Here's how the conversion works:

```python
import torch
import numpy as np

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.

    Args:
        q: [..., 4] quaternion (w, x, y, z)

    Returns:
        R: [..., 3, 3] rotation matrix
    """
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(-1)

    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=-1),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=-1),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=-1)
    ], dim=-2)

    return R

def rotation_matrix_to_quaternion(R):
    """
    Convert rotation matrix to quaternion.

    Args:
        R: [..., 3, 3] rotation matrix

    Returns:
        q: [..., 4] quaternion (w, x, y, z)
    """
    batch_shape = R.shape[:-2]

    # Compute trace
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]

    # Handle different cases based on trace
    w = torch.sqrt(torch.clamp(1 + trace, min=1e-8)) / 2
    x = (R[..., 2, 1] - R[..., 1, 2]) / (4 * w + 1e-8)
    y = (R[..., 0, 2] - R[..., 2, 0]) / (4 * w + 1e-8)
    z = (R[..., 1, 0] - R[..., 0, 1]) / (4 * w + 1e-8)

    q = torch.stack([w, x, y, z], dim=-1)
    return q / (q.norm(dim=-1, keepdim=True) + 1e-8)
```

---

## 4. Each Amino Acid Has Its Own Coordinate System

### The Frame Representation

Here's a beautiful insight that lies at the heart of RFDiffusion's architecture: **each amino acid in a protein can be described as having its own local coordinate system**.

Think about what defines the position and orientation of a single residue in a protein. The backbone atoms - nitrogen (N), alpha carbon (C$\alpha$), and carbonyl carbon (C) - form a relatively rigid unit. The position of the C$\alpha$ tells you where the residue is located in space. And the relative positions of N and C, measured from C$\alpha$, define a local coordinate frame - a set of three orthogonal axes that tell you how the residue is oriented.

This observation leads to the **frame representation**: we describe each residue $i$ as a rigid body transformation $T_i = (R_i, \vec{t}_i)$, where:
- $\vec{t}_i \in \mathbb{R}^3$ is the position of the C$\alpha$ atom (a translation)
- $R_i \in SO(3)$ is a rotation matrix defining the local coordinate system

Together, $(R_i, \vec{t}_i)$ completely specifies where the residue is and how it's oriented. This is exactly the information needed to place the backbone atoms and, given the sidechain identity, to build the full atomic structure.

The mathematical structure we're describing here is the **Special Euclidean group SE(3)**, which consists of all rigid body transformations (combinations of rotations and translations) in 3D. A transformation $T = (R, \vec{t})$ acts on a point $\vec{x}$ as:

$$T \cdot \vec{x} = R\vec{x} + \vec{t}$$

Here's an implementation of this frame representation:

```python
class RigidTransform:
    """Rigid body transformation (rotation + translation)."""

    def __init__(self, rotations, translations):
        """
        Args:
            rotations: [..., 3, 3] rotation matrices
            translations: [..., 3] translation vectors
        """
        self.rots = rotations
        self.trans = translations

    @classmethod
    def identity(cls, batch_shape, device='cpu'):
        rots = torch.eye(3, device=device).expand(*batch_shape, 3, 3).clone()
        trans = torch.zeros(*batch_shape, 3, device=device)
        return cls(rots, trans)

    def compose(self, other):
        """Compose: self @ other = (R1 R2, R1 t2 + t1)"""
        new_rots = torch.einsum('...ij,...jk->...ik', self.rots, other.rots)
        new_trans = torch.einsum('...ij,...j->...i', self.rots, other.trans) + self.trans
        return RigidTransform(new_rots, new_trans)

    def apply(self, points):
        """Apply transformation: R @ x + t"""
        return torch.einsum('...ij,...j->...i', self.rots, points) + self.trans

    def invert(self):
        """Compute inverse: (R^T, -R^T t)"""
        inv_rots = self.rots.transpose(-1, -2)
        inv_trans = -torch.einsum('...ij,...j->...i', inv_rots, self.trans)
        return RigidTransform(inv_rots, inv_trans)

    def to_tensor_7(self):
        """Convert to 7D representation (quaternion + translation)."""
        quat = rotation_matrix_to_quaternion(self.rots)
        return torch.cat([quat, self.trans], dim=-1)

    @classmethod
    def from_tensor_7(cls, tensor):
        """Create from 7D representation."""
        quat = tensor[..., :4]
        trans = tensor[..., 4:]
        rots = quaternion_to_rotation_matrix(quat)
        return cls(rots, trans)
```

Why is this representation so powerful? Because it naturally captures the SE(3) structure of the problem. When you rotate and translate an entire protein, each residue's frame transforms in a consistent way. And the relative geometry between residues - how one frame relates to another - is independent of how the protein is oriented globally.

---

## 5. Diffusion Meets Geometry: The IGSO(3) Distribution

### The Challenge of Adding Noise to Rotations

Now we come to one of the most elegant aspects of RFDiffusion: how do you run a diffusion process on something as exotic as rotations?

Recall how standard diffusion works for images or other Euclidean data. You start with your data $x_0$ and gradually add Gaussian noise:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

where $\epsilon \sim \mathcal{N}(0, I)$ is standard Gaussian noise. As $t$ increases, the signal gradually disappears into a sea of noise. A neural network then learns to reverse this process, starting from pure noise and gradually denoising to recover (or generate new) samples.

This works beautifully for translations - they live in ordinary $\mathbb{R}^3$ space where Gaussian distributions make perfect sense. But what about rotations?

Rotations don't live in Euclidean space. They live on a curved manifold called SO(3). You can't just "add" two rotations like you add vectors. And if you try to add Gaussian noise directly to the numbers in a rotation matrix, you'll end up with something that isn't a valid rotation matrix at all.

### The Intuition Behind IGSO(3)

The solution is to use a distribution that respects the geometry of SO(3). The key insight is to think about what "adding noise to a rotation" should mean intuitively.

Imagine you're holding a gyroscope perfectly level. Adding a small amount of noise should randomly perturb its orientation by a small angle. Adding more noise should perturb it by a larger angle, on average. At maximum noise, the gyroscope should be pointing in a completely random direction - uniformly distributed over all possible orientations.

This is exactly what the **Isotropic Gaussian distribution on SO(3)**, or IGSO(3), does. It's parameterized by a concentration parameter $\sigma$, and it has this beautiful property: a rotation sampled from IGSO(3) with parameter $\sigma$ rotates the identity by a random angle $\omega$ drawn from a (roughly) Gaussian distribution with standard deviation $\sigma$, around a uniformly random axis.

More precisely, the IGSO(3) distribution with concentration $\sigma$ has density proportional to:

$$p(R | \sigma) \propto \exp\left(-\frac{\omega^2}{2\sigma^2}\right)$$

where $\omega$ is the rotation angle of $R$ (the angle you'd need to rotate to go from the identity to $R$).

When $\sigma$ is small, rotations sampled from IGSO(3) are close to the identity - small random perturbations. When $\sigma$ is large, the distribution spreads out over all of SO(3). In the limit $\sigma \to \infty$, we get the uniform distribution over rotations.

Here's how to sample from this distribution:

```python
def sample_igso3(shape, sigma, device='cpu'):
    """
    Sample from Isotropic Gaussian on SO(3).

    Args:
        shape: output shape (excluding last dimensions)
        sigma: concentration parameter (higher = more noise)
        device: torch device

    Returns:
        rotations: [..., 3, 3] sampled rotation matrices
    """
    # Sample rotation angle from wrapped normal
    # For small sigma, this is approximately normal
    omega = torch.abs(torch.randn(*shape, device=device) * sigma)

    # Sample random axis uniformly on S^2
    axis = torch.randn(*shape, 3, device=device)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)

    # Convert axis-angle to rotation matrix
    rotations = axis_angle_to_rotation_matrix(axis * omega.unsqueeze(-1))

    return rotations

def axis_angle_to_rotation_matrix(axis_angle):
    """
    Convert axis-angle representation to rotation matrix using Rodrigues formula.

    Args:
        axis_angle: [..., 3] axis-angle vector (axis * angle)

    Returns:
        R: [..., 3, 3] rotation matrix
    """
    angle = axis_angle.norm(dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)

    K = skew_symmetric(axis)  # [..., 3, 3]

    # Rodrigues formula: R = I + sin(theta)K + (1-cos(theta))K^2
    I = torch.eye(3, device=axis_angle.device).expand(*axis_angle.shape[:-1], 3, 3)
    sin_angle = torch.sin(angle).unsqueeze(-1)
    cos_angle = torch.cos(angle).unsqueeze(-1)

    R = I + sin_angle * K + (1 - cos_angle) * torch.einsum('...ij,...jk->...ik', K, K)

    return R

def skew_symmetric(v):
    """Create skew-symmetric matrix from vector."""
    batch_shape = v.shape[:-1]
    zero = torch.zeros(*batch_shape, device=v.device)

    return torch.stack([
        torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], zero], dim=-1)
    ], dim=-2)
```

### The Complete SE(3) Diffusion Process

With IGSO(3) in hand, we can now define the full diffusion process for protein frames. The translation component uses standard Gaussian diffusion, while the rotation component uses IGSO(3):

```python
def diffuse_translations(translations, t, noise_schedule):
    """
    Add noise to translations using standard Gaussian diffusion.

    Args:
        translations: [..., 3] original translations
        t: [...] timestep (0 to 1)
        noise_schedule: schedule object with alpha_bar

    Returns:
        noisy_translations: [..., 3]
        noise: [..., 3] the added noise
    """
    alpha_bar = noise_schedule.alpha_bar(t)  # [..., 1]
    noise = torch.randn_like(translations)

    noisy = torch.sqrt(alpha_bar) * translations + torch.sqrt(1 - alpha_bar) * noise

    return noisy, noise


def diffuse_rotations(rotations, t, noise_schedule):
    """
    Add noise to rotations using IGSO3.

    Args:
        rotations: [..., 3, 3] original rotation matrices
        t: [...] timestep (0 to 1)
        noise_schedule: schedule with sigma(t)

    Returns:
        noisy_rotations: [..., 3, 3]
        noise_rotations: [..., 3, 3] the noise rotation applied
    """
    sigma = noise_schedule.sigma(t)  # [...]

    # Sample noise rotation from IGSO3
    noise_rot = sample_igso3(rotations.shape[:-2], sigma, rotations.device)

    # Apply noise: R_t = R_noise @ R_0
    noisy_rotations = torch.einsum('...ij,...jk->...ik', noise_rot, rotations)

    return noisy_rotations, noise_rot


class SE3DiffusionSchedule:
    """Noise schedule for SE(3) diffusion."""

    def __init__(self, T=1000, trans_sigma_max=10.0, rot_sigma_max=1.5):
        self.T = T
        self.trans_sigma_max = trans_sigma_max
        self.rot_sigma_max = rot_sigma_max

    def alpha_bar(self, t):
        """Noise schedule for translations (cosine schedule)."""
        s = 0.008
        f_t = torch.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        f_0 = np.cos(s / (1 + s) * np.pi / 2) ** 2
        return f_t / f_0

    def sigma_trans(self, t):
        """Translation noise level."""
        alpha_bar = self.alpha_bar(t)
        return torch.sqrt(1 - alpha_bar) * self.trans_sigma_max

    def sigma_rot(self, t):
        """Rotation noise level (linear in t)."""
        return t * self.rot_sigma_max

    def sample_timestep(self, batch_size, device='cpu'):
        """Sample random timesteps."""
        return torch.rand(batch_size, device=device)


def diffuse_frames(frames, t, schedule):
    """
    Diffuse SE(3) frames.

    Args:
        frames: RigidTransform object
        t: [...] timesteps
        schedule: SE3DiffusionSchedule

    Returns:
        noisy_frames: RigidTransform
        noise: dict with 'trans' and 'rot' noise
    """
    # Diffuse translations
    noisy_trans, trans_noise = diffuse_translations(
        frames.trans, t.unsqueeze(-1), schedule
    )

    # Diffuse rotations
    noisy_rots, rot_noise = diffuse_rotations(
        frames.rots, t, schedule
    )

    noisy_frames = RigidTransform(noisy_rots, noisy_trans)
    noise = {'trans': trans_noise, 'rot': rot_noise}

    return noisy_frames, noise
```

---

## 6. Building Equivariant Neural Networks

### The Art of Computing Without Breaking Symmetry

Now that we understand how to add noise to protein frames, the next question is: how do we build a neural network that can denoise them while respecting SE(3) equivariance?

The fundamental challenge is this: standard neural network operations - linear layers, convolutions, attention - don't know anything about 3D geometry. If you feed a rotated input into a standard network, you'll get a completely different output, not a rotated version of the original output.

The solution lies in carefully designing network operations that respect the symmetry. There are two complementary strategies:

**Invariant features** are quantities that don't change at all under SE(3) transformations. Distances between atoms, angles between bonds, and dihedral angles are all invariant - they're properties of the intrinsic geometry of the molecule, independent of how it's oriented in space. These features are "safe" to use anywhere in the network because they carry no orientation information that could break equivariance.

**Equivariant operations** work directly with vectors and frames but transform them consistently. The key principle is: compute messages and updates using invariant features, then apply the results in local coordinate frames so that global transformations propagate correctly.

### Building an Equivariant Layer

Here's a concrete example of an SE(3) equivariant convolution layer:

```python
class SE3EquivariantConv(nn.Module):
    """SE(3) equivariant convolution layer."""

    def __init__(self, node_dim, edge_dim, hidden_dim=128):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim

        # Edge feature computation (invariant)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Message computation
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        # Position update (in local frame)
        self.pos_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3D displacement
        )

    def forward(self, node_features, frames, edge_index, edge_features):
        """
        Args:
            node_features: [N, node_dim] node features
            frames: RigidTransform [N] frames
            edge_index: [2, E] edge indices
            edge_features: [E, edge_dim] invariant edge features

        Returns:
            updated_node_features: [N, node_dim]
            position_updates: [N, 3] equivariant position updates
        """
        src, dst = edge_index
        N = node_features.shape[0]

        # Compute edge embeddings
        edge_emb = self.edge_mlp(edge_features)  # [E, hidden]

        # Compute messages
        src_features = node_features[src]  # [E, node_dim]
        dst_features = node_features[dst]  # [E, node_dim]

        message_input = torch.cat([src_features, dst_features, edge_emb], dim=-1)
        messages = self.message_mlp(message_input)  # [E, node_dim]

        # Aggregate messages
        aggregated = torch.zeros(N, self.node_dim, device=messages.device)
        aggregated.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.node_dim), messages)

        # Compute position updates (in source frame, then transform to global)
        pos_updates_local = self.pos_mlp(message_input)  # [E, 3]

        # Transform to global frame using source frame
        src_frames = RigidTransform(frames.rots[src], frames.trans[src])
        pos_updates_global = src_frames.apply(pos_updates_local) - src_frames.trans

        # Aggregate position updates
        pos_aggregated = torch.zeros(N, 3, device=pos_updates_global.device)
        pos_aggregated.scatter_add_(0, dst.unsqueeze(-1).expand(-1, 3), pos_updates_global)

        return node_features + aggregated, pos_aggregated
```

The key insight is in the position updates. The network predicts a 3D displacement *in the local coordinate frame of each source residue*. This local displacement is then transformed to global coordinates using the source frame's rotation. Because the network only "sees" invariant features and outputs local displacements, the whole operation is equivariant: if you rotate the input, the output rotates in exactly the same way.

### Computing Invariant Edge Features

The edge features that feed into our network must be SE(3)-invariant. Here's how to compute them from the frame representation:

```python
def compute_invariant_edge_features(frames, edge_index, max_dist=20.0):
    """
    Compute SE(3)-invariant edge features.

    Args:
        frames: RigidTransform [N]
        edge_index: [2, E]
        max_dist: maximum distance for RBF encoding

    Returns:
        edge_features: [E, edge_dim]
    """
    src, dst = edge_index

    # Pairwise distances (invariant)
    pos_src = frames.trans[src]
    pos_dst = frames.trans[dst]
    distances = (pos_src - pos_dst).norm(dim=-1, keepdim=True)

    # RBF encoding of distance
    rbf_dist = rbf_encode(distances, n_bins=16, max_val=max_dist)

    # Relative orientation (invariant)
    # R_rel = R_dst^T @ R_src
    R_rel = torch.einsum('eij,ejk->eik',
                         frames.rots[dst].transpose(-1, -2),
                         frames.rots[src])

    # Extract invariants from relative rotation
    # Trace is invariant
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    rot_angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))

    # Combine features
    edge_features = torch.cat([
        rbf_dist,
        rot_angle.unsqueeze(-1),
        torch.sin(rot_angle).unsqueeze(-1),
        torch.cos(rot_angle).unsqueeze(-1)
    ], dim=-1)

    return edge_features

def rbf_encode(distances, n_bins=16, max_val=20.0):
    """Radial basis function encoding of distances."""
    centers = torch.linspace(0, max_val, n_bins, device=distances.device)
    gamma = 1.0 / (max_val / n_bins)
    return torch.exp(-gamma * (distances - centers) ** 2)
```

Notice what features we're computing: distances (obviously invariant), and the relative rotation angle between frames. The relative rotation between two frames is independent of how the whole protein is oriented - it's an intrinsic property of their geometric relationship.

---

## 7. The RFDiffusion Architecture

### Standing on the Shoulders of Giants

RFDiffusion doesn't build its architecture from scratch. Instead, it cleverly adapts the architecture of RoseTTAFold, a highly successful protein structure prediction model, for the generative task. This is a recurring theme in modern ML: pretrained representations and architectures from one task often transfer remarkably well to related tasks.

The overall structure looks like this:

```
+------------------------------------------------------------------+
|                         RFDiffusion                               |
+------------------------------------------------------------------+
|                                                                   |
|  +-----------+    +---------------+    +-----------------+        |
|  |  Noisy    |    |    SE(3)      |    |    Frame        |        |
|  |  Frames   |--->|  Transformer  |--->|    Updates      |        |
|  |  + Cond.  |    |  (RoseTTA-    |    |    (Denoise)    |        |
|  +-----------+    |    Fold)      |    +-----------------+        |
|                   +---------------+                               |
|                                                                   |
+------------------------------------------------------------------+
```

The model takes noisy frames (plus any conditioning information) and predicts updates to those frames that move them closer to clean protein structures. This is repeated over many diffusion steps, gradually denoising random noise into coherent protein backbones.

### A Single RFDiffusion Block

Here's what one block of the architecture looks like:

```python
class RFDiffusionBlock(nn.Module):
    """Single block of RFDiffusion."""

    def __init__(self, node_dim=256, pair_dim=128, n_heads=8):
        super().__init__()

        # Node update (IPA-like)
        self.node_attn = InvariantPointAttention(node_dim, pair_dim, n_heads)

        # Pair update (triangular)
        self.pair_update = TriangularUpdate(pair_dim)

        # Frame update
        self.frame_update = FrameUpdateLayer(node_dim)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(256, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim)
        )

    def forward(self, node_features, pair_features, frames, t_embed):
        """
        Args:
            node_features: [L, node_dim]
            pair_features: [L, L, pair_dim]
            frames: RigidTransform [L]
            t_embed: [256] timestep embedding

        Returns:
            updated tensors
        """
        # Add timestep information
        time_bias = self.time_embed(t_embed)
        node_features = node_features + time_bias.unsqueeze(0)

        # Node attention with IPA
        node_features = self.node_attn(node_features, pair_features, frames)

        # Pair update
        pair_features = self.pair_update(pair_features)

        # Frame update
        frame_updates = self.frame_update(node_features)
        frames = update_frames(frames, frame_updates)

        return node_features, pair_features, frames
```

Several components work together:
- **Invariant Point Attention (IPA)**: A specialized attention mechanism (borrowed from AlphaFold2) that attends over residues while respecting SE(3) equivariance.
- **Triangular updates**: Operations on pair features that maintain consistency across the 2D matrix of pairwise relationships.
- **Frame updates**: The output layer that predicts how to update each residue's frame.
- **Timestep embedding**: The model needs to know what noise level it's denoising from, so we embed the timestep and add it to the features.

### Predicting Frame Updates

The heart of the denoising process is predicting how to update each frame:

```python
class FrameUpdateLayer(nn.Module):
    """Predict frame updates (rotation + translation)."""

    def __init__(self, node_dim):
        super().__init__()

        self.norm = nn.LayerNorm(node_dim)

        # Predict 6D representation: 3 for rotation, 3 for translation
        self.to_update = nn.Linear(node_dim, 6)

        # Scale factors
        self.rot_scale = 0.1
        self.trans_scale = 1.0

    def forward(self, node_features):
        """
        Args:
            node_features: [L, node_dim]

        Returns:
            frame_updates: dict with 'rot' and 'trans'
        """
        x = self.norm(node_features)
        updates = self.to_update(x)

        rot_update = updates[:, :3] * self.rot_scale
        trans_update = updates[:, 3:] * self.trans_scale

        return {'rot': rot_update, 'trans': trans_update}

def update_frames(frames, updates):
    """Apply predicted updates to frames."""
    # Create update transformation
    rot_update = axis_angle_to_rotation_matrix(updates['rot'])
    trans_update = updates['trans']

    update_transform = RigidTransform(rot_update, trans_update)

    # Compose: new_frame = old_frame @ update
    return frames.compose(update_transform)
```

The network outputs small updates in axis-angle format for rotations and Cartesian coordinates for translations. These updates are applied through composition with the current frames, gradually refining the structure.

---

## 8. Conditional Generation: Where the Magic Happens

### Beyond Random Generation

Generating random protein structures is impressive, but the real power of RFDiffusion lies in **conditional generation** - the ability to generate proteins that satisfy specific constraints or design objectives. This is where the practical applications emerge: designing proteins that bind to specific targets, building scaffolds around functional motifs, creating symmetric assemblies.

RFDiffusion supports several types of conditioning:

**Motif scaffolding**: You have a specific arrangement of amino acids that performs some function - maybe a binding site or a catalytic center - and you want to generate a protein that holds those residues in exactly the right positions. RFDiffusion can generate the surrounding scaffold while keeping the motif fixed.

**Binder design**: Given a target protein, generate a new protein that binds to a specific surface on the target. This is the key to designing therapeutic antibodies, protein inhibitors, and biosensors.

**Symmetric assemblies**: Generate proteins with specific symmetry properties - dimers, trimers, or higher-order assemblies that could form cages, rings, or other regular structures.

**Secondary structure conditioning**: Specify that you want a certain pattern of helices and sheets, and let the model figure out the 3D arrangement.

### Motif Conditioning in Practice

The simplest and most powerful form of conditioning is motif scaffolding. The idea is straightforward: at each step of the diffusion process, replace the noisy frames at motif positions with the ground truth (or lightly noised versions of it). The model learns to generate scaffolds that are compatible with the fixed motif.

```python
class MotifConditioning:
    """Condition on fixed motif residues."""

    def __init__(self, motif_positions, motif_frames):
        """
        Args:
            motif_positions: list of residue indices that are fixed
            motif_frames: RigidTransform for motif residues
        """
        self.motif_pos = set(motif_positions)
        self.motif_frames = motif_frames

    def apply(self, frames, t):
        """
        Replace motif positions with ground truth.

        In practice, the noise level for motif is set to 0.
        """
        for i, pos in enumerate(self.motif_pos):
            frames.rots[pos] = self.motif_frames.rots[i]
            frames.trans[pos] = self.motif_frames.trans[i]

        return frames

    def get_conditioning_mask(self, L):
        """Return mask where 1 = motif (fixed), 0 = scaffold (generate)."""
        mask = torch.zeros(L)
        for pos in self.motif_pos:
            mask[pos] = 1
        return mask
```

This simple approach is remarkably effective. By fixing certain residues, we constrain the generative process to produce only structures compatible with those constraints. The model has learned, through training on thousands of protein structures, what kinds of scaffolds are geometrically and physically plausible.

### Self-Conditioning: Learning from Your Own Predictions

RFDiffusion uses a clever technique called **self-conditioning** that significantly improves sample quality. The idea is simple: feed the model's previous prediction back as additional input to the next step.

```python
class SelfConditionedRFDiffusion(nn.Module):
    """RFDiffusion with self-conditioning."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.self_cond_proj = nn.Linear(7, 128)  # 4 quat + 3 trans

    def forward(self, noisy_frames, t, prev_pred=None):
        """
        Args:
            noisy_frames: current noisy frames
            t: timestep
            prev_pred: previous model prediction (optional)
        """
        # Encode self-conditioning
        if prev_pred is not None:
            self_cond = self.self_cond_proj(prev_pred.to_tensor_7())
        else:
            self_cond = torch.zeros(noisy_frames.trans.shape[0], 128)

        # Run model with self-conditioning
        return self.base_model(noisy_frames, t, self_cond)
```

Why does this help? Diffusion models face a challenging task: given only the current noisy state, predict the clean data. But at each step, the model has access to additional information - its own previous prediction. By conditioning on this prediction, the model can maintain consistency across steps and make more informed updates.

### Classifier-Free Guidance: Turning Up the Conditioning

Sometimes you want stronger conditioning - you want the model to really commit to satisfying your constraints, even at the cost of some diversity. **Classifier-free guidance** provides a knob to control this trade-off.

The idea is to train the model to work both with and without conditioning (by randomly dropping the conditioning during training). At inference time, we compute two predictions - one with conditioning and one without - and extrapolate in the direction of conditioning:

```python
def sample_with_guidance(model, initial_frames, condition, guidance_scale=2.0, steps=100):
    """
    Sample with classifier-free guidance.

    Args:
        model: RFDiffusion model
        initial_frames: starting noisy frames
        condition: conditioning information
        guidance_scale: strength of conditioning
        steps: number of denoising steps
    """
    frames = initial_frames
    schedule = SE3DiffusionSchedule()

    for step in range(steps, 0, -1):
        t = torch.tensor([step / steps])

        # Conditional prediction
        pred_cond = model(frames, t, condition)

        # Unconditional prediction
        pred_uncond = model(frames, t, None)

        # Guided prediction
        pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

        # Denoise step
        frames = denoise_step(frames, pred, t, schedule)

    return frames
```

When `guidance_scale = 1`, we use pure conditional prediction. When `guidance_scale > 1`, we push further in the direction of conditioning, making the generated structures more likely to satisfy the constraints at the cost of reduced diversity.

---

## 9. Training the Model

### Learning to Denoise

The training objective for RFDiffusion is conceptually simple: given a noisy protein structure and the noise level, predict the clean structure. The loss is computed separately for the translation and rotation components:

```python
def rfdiffusion_loss(pred_frames, true_frames, t, loss_weights=None):
    """
    Compute RFDiffusion training loss.

    Args:
        pred_frames: predicted clean frames
        true_frames: ground truth frames
        t: timestep
    """
    losses = {}

    # Translation loss (MSE)
    trans_loss = ((pred_frames.trans - true_frames.trans) ** 2).sum(dim=-1).mean()
    losses['trans'] = trans_loss

    # Rotation loss (geodesic distance on SO(3))
    R_diff = torch.einsum('...ij,...kj->...ik',
                          pred_frames.rots,
                          true_frames.rots)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    rot_angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))
    rot_loss = (rot_angle ** 2).mean()
    losses['rot'] = rot_loss

    # Combined loss
    if loss_weights is None:
        loss_weights = {'trans': 1.0, 'rot': 1.0}

    total_loss = sum(loss_weights.get(k, 1.0) * v for k, v in losses.items())

    return total_loss, losses
```

For translations, we use standard mean squared error - the L2 distance in Euclidean space. For rotations, we use the geodesic distance on SO(3), which is the angle you'd need to rotate to go from one orientation to the other. This is the natural metric on the rotation manifold, and using it ensures the loss correctly measures rotational error.

A complete training step looks like:

```python
def training_step(model, batch, optimizer, schedule):
    """Single training step."""
    true_frames = batch['frames']  # Ground truth
    L = true_frames.trans.shape[0]

    # Sample timestep
    t = schedule.sample_timestep(1).expand(L)

    # Add noise
    noisy_frames, noise = diffuse_frames(true_frames, t, schedule)

    # Predict clean frames
    pred_frames = model(noisy_frames, t)

    # Compute loss
    loss, loss_dict = rfdiffusion_loss(pred_frames, true_frames, t)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss_dict
```

---

## 10. Generating New Proteins

### The Denoising Dance

Generation is the reverse of training: we start from pure noise and gradually denoise to produce a clean structure. This is where months of training pay off in seconds of generation.

```python
@torch.no_grad()
def sample_rfdiffusion(model, L, schedule, n_steps=100, condition=None):
    """
    Sample a protein backbone using RFDiffusion.

    Args:
        model: trained RFDiffusion model
        L: sequence length
        schedule: noise schedule
        n_steps: number of denoising steps
        condition: optional conditioning

    Returns:
        frames: generated RigidTransform
    """
    # Initialize with random noise
    frames = RigidTransform.identity((L,), device='cuda')

    # Add maximum noise
    t_max = torch.ones(L, device='cuda')
    frames, _ = diffuse_frames(frames, t_max, schedule)

    # Self-conditioning
    prev_pred = None

    # Denoise
    timesteps = torch.linspace(1, 0, n_steps + 1)

    for i in range(n_steps):
        t_now = timesteps[i]
        t_next = timesteps[i + 1]

        t_tensor = torch.full((L,), t_now, device='cuda')

        # Model prediction
        pred_frames = model(frames, t_tensor, condition, prev_pred)

        # Self-conditioning: 50% chance to use prediction
        if torch.rand(1) < 0.5:
            prev_pred = pred_frames

        # Compute denoised estimate and re-noise to t_next
        if t_next > 0:
            frames = denoise_and_renoise(frames, pred_frames, t_now, t_next, schedule)
        else:
            frames = pred_frames

        # Apply conditioning (e.g., fix motif positions)
        if condition is not None:
            frames = condition.apply(frames, t_next)

    return frames
```

The denoising step is carefully designed to work on SE(3):

```python
def denoise_and_renoise(noisy_frames, pred_clean, t_now, t_next, schedule):
    """
    Denoise from t_now to t_next.

    Implements the DDPM update rule adapted for SE(3).
    """
    # For translations: standard DDPM
    alpha_now = schedule.alpha_bar(torch.tensor(t_now))
    alpha_next = schedule.alpha_bar(torch.tensor(t_next))

    # Estimate noise
    noise_est = (noisy_frames.trans - torch.sqrt(alpha_now) * pred_clean.trans) / torch.sqrt(1 - alpha_now)

    # Predict x_{t-1}
    trans_mean = torch.sqrt(alpha_next) * pred_clean.trans + torch.sqrt(1 - alpha_next) * noise_est

    # Add noise (unless t_next = 0)
    if t_next > 0:
        noise = torch.randn_like(trans_mean)
        beta = 1 - alpha_next / alpha_now
        trans_next = trans_mean + torch.sqrt(beta) * noise
    else:
        trans_next = trans_mean

    # For rotations: interpolate on SO(3)
    sigma_now = schedule.sigma_rot(torch.tensor(t_now))
    sigma_next = schedule.sigma_rot(torch.tensor(t_next))

    # Interpolate toward predicted rotation
    interp_factor = sigma_next / (sigma_now + 1e-8)
    rots_next = slerp_rotation(noisy_frames.rots, pred_clean.rots, 1 - interp_factor)

    return RigidTransform(rots_next, trans_next)
```

For translations, we use the standard DDPM update rule. For rotations, we interpolate on the SO(3) manifold using spherical linear interpolation (SLERP) between the current noisy rotation and the predicted clean rotation.

---

## 11. Real Success Stories

### Proteins That Actually Work

The ultimate test of any protein design method is experimental validation: do the designed proteins actually fold? Do they perform their intended function?

RFDiffusion has passed this test with flying colors. In the original Nature paper, the authors demonstrated several remarkable achievements:

**Novel folds**: RFDiffusion can generate protein backbones with fold topologies never seen in nature. These aren't just theoretically interesting - when the corresponding sequences were synthesized and expressed, the proteins folded into the predicted structures with remarkable accuracy.

**Symmetric assemblies**: By incorporating symmetry constraints into the generation process, RFDiffusion can design proteins that assemble into rings, cages, and other symmetric structures. This opens possibilities for drug delivery, vaccine design, and nanotechnology.

**Functional binders**: Perhaps most impressively, RFDiffusion has been used to design proteins that bind to specific targets - including therapeutically relevant proteins - with high affinity and specificity. Combined with ProteinMPNN for sequence design, this creates a complete pipeline from target specification to functional protein.

**Enzyme scaffolds**: By scaffolding around known catalytic motifs, researchers have used RFDiffusion to create novel enzymes. The model figures out how to build a protein that holds the catalytic residues in exactly the right positions for activity.

These results represent a qualitative leap beyond previous protein design methods. RFDiffusion doesn't just tweak existing proteins - it generates genuinely new structures that evolution never explored.

---

## 12. The Bigger Picture: A Comparison

How does RFDiffusion compare to other structure generation methods?

| Method | Representation | Equivariance | Generation Approach |
|--------|---------------|--------------|---------------------|
| RFDiffusion | SE(3) frames | Full SE(3) | Diffusion |
| Chroma | Distance matrix | E(3) invariant | Diffusion |
| FrameDiff | SE(3) frames | SE(3) | Flow matching |
| Genie | Backbone angles | None | Autoregressive |

RFDiffusion stands out for its combination of SE(3) equivariance, rich conditioning mechanisms, and the powerful pretrained representations inherited from RoseTTAFold. Other methods make different trade-offs: Chroma uses an invariant representation (distance matrices) which is simpler but loses some geometric information; FrameDiff uses flow matching instead of diffusion; Genie generates autoregressively without explicit geometric equivariance.

Each approach has its strengths, and the field is evolving rapidly. But RFDiffusion's success has established SE(3) diffusion as a dominant paradigm for protein generation.

---

## 13. Key Takeaways

Let's step back and summarize the key ideas from this lecture:

**Proteins are geometric objects.** Their properties depend on shape, not orientation. This makes SE(3) equivariance a natural requirement for protein generative models.

**Each residue has its own coordinate system.** The frame representation - treating each amino acid as a rigid body transformation - elegantly captures the geometry of protein backbones.

**Diffusion on rotations requires special care.** Standard Gaussian diffusion doesn't work for rotations, which live on a curved manifold. The IGSO(3) distribution provides a principled way to add and remove noise on SO(3).

**Equivariance is built into the architecture.** By using invariant features for computation and applying updates in local frames, we can build neural networks that respect SE(3) symmetry by construction.

**Conditioning is where practical applications emerge.** Motif scaffolding, binder design, and symmetric assembly generation all work by constraining the diffusion process to satisfy specific objectives.

**RFDiffusion generates proteins that actually work.** Experimental validation has confirmed that RFDiffusion-designed proteins fold as predicted and perform intended functions.

The ability to generate novel protein structures on demand - structures that actually fold and function - represents a revolution in biotechnology. For the first time, we can envision designing proteins for specific applications without being limited to what nature has provided. RFDiffusion is a major step toward this future, combining beautiful mathematics with practical impact.

---

## 14. Further Reading

1. Watson et al. (2023). "De novo design of protein structure and function with RFdiffusion." *Nature*. The original RFDiffusion paper with extensive experimental validation.

2. Yim et al. (2023). "SE(3) diffusion model with application to protein backbone generation." *ICML*. A theoretical analysis of SE(3) diffusion and the FrameDiff model.

3. Leach et al. (2022). "Denoising Diffusion Probabilistic Models on SO(3)." *ICLR Workshop*. Foundational work on diffusion over rotation manifolds.

4. Baek et al. (2021). "Accurate prediction of protein structures and interactions using a three-track neural network." *Science*. The RoseTTAFold paper, whose architecture forms the backbone of RFDiffusion.

5. Dauparas et al. (2022). "Robust deep learning-based protein sequence design using ProteinMPNN." *Science*. The companion tool for designing sequences that fold into RFDiffusion-generated structures.
