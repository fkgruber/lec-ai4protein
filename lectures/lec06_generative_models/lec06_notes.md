# Lecture 6: Generative Models - VAEs and Diffusion

## Dreaming Up New Proteins

What if we could dream up new proteins? Not just analyze the ones that evolution has painstakingly crafted over billions of years, but actually imagine entirely new molecular machines that have never existed before. This is the promise of generative models in protein science, and it represents one of the most exciting frontiers in computational biology today.

Think about it this way: evolution has explored only a tiny fraction of the vast space of possible proteins. The proteins we see in nature today are the survivors of a brutal selection process, optimized for the particular environments and challenges their host organisms faced. But what about all the proteins that could exist, proteins that might cure diseases, break down plastic pollution, or catalyze chemical reactions more efficiently than anything nature has produced?

Generative models give us a way to explore this uncharted territory. Instead of waiting for evolution to stumble upon useful proteins through random mutation and selection, we can train machine learning models to understand the underlying patterns that make proteins work. Once these models have internalized what makes a protein a protein, they can generate entirely new sequences that obey the same fundamental rules, sequences that fold into stable structures and potentially perform useful functions.

In this lecture, we will explore two foundational approaches to generative modeling: Variational Autoencoders (VAEs) and Diffusion Models. These are not just theoretical curiosities. VAEs have been used to design novel enzymes, while diffusion models power RFDiffusion, one of the most successful protein design tools ever created. By the end of this chapter, you will understand how these models work at a conceptual and mathematical level, and you will be ready to apply them to your own protein design challenges.

---

## Part 1: Variational Autoencoders - Compressing Protein Information

### The Compression Perspective

Before we dive into the mathematics of VAEs, let us build some intuition by thinking about data compression. Imagine you have a collection of thousands of protein sequences from the same family, say, all known serine proteases. Despite their sequence diversity, these proteins share deep similarities: they all have similar folds, they all catalyze the same type of reaction, and they all have conserved residues in key positions.

Now imagine trying to describe any one of these proteins. You could list out all 200+ amino acids, but that seems wasteful. These proteins are clearly not random sequences. There must be some more compact way to represent them, some underlying code that captures the essential features that make a serine protease a serine protease.

This is exactly what autoencoders try to do. An autoencoder is a neural network that learns to compress data into a low-dimensional representation (called the latent code or latent vector) and then decompress it back to the original form. The magic happens in the middle: if the autoencoder can successfully reconstruct proteins from their compressed representations, then those representations must capture something meaningful about what makes proteins work.

But there is a problem with standard autoencoders. They learn a deterministic mapping from inputs to latent codes. Ask for the latent code of a particular protein, and you get a single point in latent space. This is fine for compression, but what if we want to generate new proteins? We could try picking random points in latent space and decoding them, but there is no guarantee that these points correspond to valid proteins. The latent space might have gaps, discontinuities, or regions that decode into gibberish.

### From Compression to Generation

Variational Autoencoders solve this problem by learning a probabilistic latent space. Instead of mapping each protein to a single point, a VAE maps it to a probability distribution. Specifically, for each input protein, the encoder outputs the parameters of a Gaussian distribution: a mean vector and a variance vector. We then sample from this distribution to get our latent code.

Why does this help? Because we also add a special regularization term that encourages these distributions to be close to a standard normal distribution (zero mean, unit variance). This has a beautiful consequence: the entire latent space becomes meaningful. If we sample a random point from a standard normal distribution and decode it, we are likely to get a valid protein, because the VAE has been trained to make this work.

Let us make this concrete with some notation. We have:
- An input protein sequence $x$
- An encoder network that takes $x$ and outputs distribution parameters $\mu_\phi(x)$ and $\sigma_\phi(x)$
- A latent code $z$ sampled from $\mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x)I)$
- A decoder network that takes $z$ and reconstructs the protein

The subscript $\phi$ reminds us that these are neural networks with learnable parameters. The encoder defines what we call the approximate posterior $q_\phi(z|x)$, which tells us: given this protein, what latent codes are likely to have generated it?

Here is what the encoder looks like in code:

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)  # log(variance) for numerical stability
        return mu, logvar
```

Notice that we output log-variance rather than variance directly. This is a practical trick that ensures the variance is always positive (since we will exponentiate it later) and makes optimization more stable.

The decoder is simpler. It takes the latent code and predicts the probability distribution over amino acids at each position:

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        return self.fc2(h)  # outputs logits for each amino acid
```

### What Are We Really Trying to Do? Motivating the ELBO

Now comes the crucial question: how do we train this thing? What objective should we optimize?

Ideally, we would like to maximize the probability of our training data under the model. In probabilistic terms, we want to maximize the log-likelihood:

$$\log p_\theta(x) = \log \int p_\theta(x|z)p(z)dz$$

This equation says: to compute the probability of a protein $x$, we need to consider all possible latent codes $z$ that could have generated it. For each $z$, we compute how likely it is under our prior $p(z)$ (which we set to be a standard normal), and how likely the protein is given that $z$ (which our decoder tells us). Then we integrate over all possible $z$ values.

This integral is intractable. There are infinitely many possible latent codes, and for most of them, we would need to run our decoder to evaluate $p_\theta(x|z)$. We need a different approach.

This is where variational inference comes in. The key insight is that we can derive a lower bound on the log-likelihood that is tractable to compute and optimize. This lower bound is called the Evidence Lower Bound, or ELBO.

The ELBO has a beautiful decomposition into two terms that each have an intuitive meaning:

$$\text{ELBO} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

Let us understand what each term means:

**Reconstruction Loss** (the first term): This measures how well we can reconstruct the input $x$ from samples of the latent code $z$. We sample $z$ from our encoder's distribution $q_\phi(z|x)$, pass it through the decoder, and measure how probable the original $x$ is under the decoder's output distribution. If the reconstruction is good, this term is high (less negative).

**KL Divergence** (the second term): This measures how different our encoder's distribution is from the prior. Remember, we want the latent space to be well-behaved, with the encoded distributions close to standard normal. The KL divergence penalizes distributions that deviate from this ideal. If the encoder outputs something close to $\mathcal{N}(0, I)$, this term is small.

The ELBO is a lower bound, meaning $\log p_\theta(x) \geq \text{ELBO}$. By maximizing the ELBO, we push up on the true log-likelihood from below.

### The ELBO Derivation

For the mathematically curious, let us see where this formula comes from. We start with a clever trick: multiplying and dividing by our encoder distribution.

$$\log p_\theta(x) = \log \int p_\theta(x|z)p(z)dz = \log \int \frac{p_\theta(x|z)p(z)}{q_\phi(z|x)}q_\phi(z|x)dz$$

Now we can apply Jensen's inequality. Since $\log$ is a concave function, we have:

$$\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$$

Applying this:

$$\log p_\theta(x) \geq \int q_\phi(z|x) \log \frac{p_\theta(x|z)p(z)}{q_\phi(z|x)}dz$$

Expanding the logarithm:

$$= \int q_\phi(z|x) \log p_\theta(x|z)dz + \int q_\phi(z|x) \log \frac{p(z)}{q_\phi(z|x)}dz$$

The first integral is our reconstruction term. The second integral is the negative KL divergence (by definition of KL divergence). Rearranging:

$$= \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

### Computing the KL Divergence

One beautiful property of choosing Gaussian distributions is that the KL divergence has a closed-form solution. When both $q_\phi(z|x)$ and $p(z)$ are Gaussian, we do not need to estimate the KL term through sampling. Instead:

$$D_{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = -\frac{1}{2}\sum_{j=1}^{J}(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

In code:

```python
def kl_divergence(mu, logvar):
    """
    KL divergence between N(mu, sigma) and N(0, 1)

    Args:
        mu: Mean of approximate posterior [batch_size, latent_dim]
        logvar: Log variance of approximate posterior [batch_size, latent_dim]

    Returns:
        KL divergence [batch_size]
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
```

### The Reparameterization Trick: Making Sampling Differentiable

We have a loss function (the negative ELBO) that we want to minimize using gradient descent. But there is a problem: the reconstruction term requires sampling $z$ from $q_\phi(z|x)$, and sampling is not a differentiable operation.

To see why this is a problem, imagine computing the gradient of the reconstruction loss with respect to the encoder parameters $\phi$. These parameters affect the mean $\mu$ and variance $\sigma^2$ of the distribution we sample from. But the sampling itself involves random number generation. How do we backpropagate through a coin flip?

The reparameterization trick provides an elegant solution. Instead of sampling directly from $\mathcal{N}(\mu, \sigma^2)$, we sample from a standard normal $\epsilon \sim \mathcal{N}(0, 1)$ and then transform:

$$z = \mu + \sigma \odot \epsilon$$

The key insight is that the randomness now comes entirely from $\epsilon$, which has no learnable parameters. The transformation from $\epsilon$ to $z$ is a deterministic function of $\mu$ and $\sigma$, so gradients can flow through it normally.

Think of it this way: instead of asking "what is the gradient of sampling?", we are asking "what is the gradient of scaling and shifting?", which is straightforward.

```python
def reparameterize(mu, logvar):
    """
    Reparameterization trick: z = mu + sigma * epsilon

    Args:
        mu: Mean of approximate posterior
        logvar: Log variance of approximate posterior

    Returns:
        Sampled latent variable z
    """
    std = torch.exp(0.5 * logvar)  # sigma = exp(log(sigma^2)/2)
    eps = torch.randn_like(std)     # epsilon ~ N(0, I)
    return mu + eps * std
```

### Putting It All Together: The Complete VAE

Now we can assemble a complete VAE for protein sequences. The model takes a protein sequence, encodes it to latent distribution parameters, samples a latent code using the reparameterization trick, and decodes back to amino acid probabilities.

```python
class ProteinVAE(nn.Module):
    def __init__(self, seq_len, vocab_size=21, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        input_dim = seq_len * vocab_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        # x: [batch, seq_len] -> one-hot: [batch, seq_len * vocab_size]
        x_onehot = nn.functional.one_hot(x, self.vocab_size).float()
        x_flat = x_onehot.view(x.size(0), -1)
        h = self.encoder(x_flat)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.decoder(z)
        return h.view(-1, self.seq_len, self.vocab_size)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def sample(self, n_samples, device='cpu'):
        z = torch.randn(n_samples, self.fc_mu.out_features).to(device)
        logits = self.decode(z)
        return torch.argmax(logits, dim=-1)
```

The loss function combines reconstruction and KL terms:

```python
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction Loss + beta * KL Divergence
    """
    # Reconstruction loss (cross-entropy for sequences)
    recon_loss = nn.functional.cross_entropy(
        recon_x.view(-1, recon_x.size(-1)),
        x.view(-1),
        reduction='mean'
    )

    # KL divergence
    kl_loss = kl_divergence(mu, logvar).mean()

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss
```

Notice the $\beta$ parameter. This is from a variant called $\beta$-VAE, where increasing $\beta$ encourages a more structured latent space at the cost of reconstruction quality. This trade-off can be useful for learning disentangled representations, where different dimensions of the latent space control different properties of the protein.

---

## Part 2: Diffusion Models - Learning to Reverse Corruption

### A Different Philosophy of Generation

While VAEs learn to compress data into a structured latent space, diffusion models take a completely different approach. Imagine slowly corrupting a protein structure by adding random noise, step by step, until it becomes indistinguishable from pure static. Now imagine learning to reverse that process, to take pure noise and gradually sculpt it into a realistic protein.

This is the core idea behind diffusion models, and it has proven remarkably effective. Diffusion models now produce the highest-quality samples across many domains, from images to audio to protein structures.

The intuition is appealing: denoising is easier than generating from scratch. If I show you a slightly fuzzy image and ask you to clean it up, that is much easier than asking you to imagine a new image from nothing. Diffusion models exploit this by breaking generation into many small, manageable denoising steps.

### The Forward Process: Controlled Destruction

The forward process defines how we corrupt data. Starting from a clean data point $x_0$, we add Gaussian noise over $T$ timesteps to produce increasingly noisy versions $x_1, x_2, \ldots, x_T$. By the final step, $x_T$ should be essentially pure noise.

At each step, we have:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

The parameter $\beta_t$ controls how much noise we add at step $t$. This is called the noise schedule, and it is typically designed to start small (gentle corruption at early steps) and increase over time (aggressive corruption at later steps).

Here is a crucial mathematical insight: we do not need to apply noise sequentially. There is a closed-form formula that lets us jump directly from $x_0$ to any $x_t$:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$ is the cumulative product.

This means we can sample noisy data at any timestep in one shot:

$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This property is essential for efficient training, as we will see shortly.

```python
class DiffusionSchedule:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, T)

        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def add_noise(self, x0, t, noise=None):
        """
        Sample x_t from q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1)

        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
```

### The Reverse Process: Learning to Denoise

The forward process is fixed and known. The magic happens in the reverse process, where we learn a neural network to undo the corruption.

The reverse process starts from pure noise $x_T \sim \mathcal{N}(0, I)$ and iteratively denoises to recover the data:

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

The neural network predicts the mean of this distribution. But what exactly should it predict? It turns out that instead of predicting the mean directly, it is easier to predict the noise that was added. The mean can then be computed as:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)$$

This reformulation is not just a mathematical convenience. It makes the training objective much simpler.

### The Training Objective: Just Predict the Noise

The training objective for diffusion models is beautifully simple. Given a clean data point $x_0$, we:
1. Sample a random timestep $t$
2. Sample random noise $\epsilon$
3. Compute the noisy version $x_t$
4. Ask the network to predict the noise
5. Minimize the mean squared error between predicted and true noise

$$\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

That is it. The network learns to look at a noisy sample and figure out what noise was added. This simple objective, proposed by Ho et al. in 2020, works remarkably well.

```python
def diffusion_loss(model, x0, schedule):
    """
    Compute diffusion training loss
    """
    batch_size = x0.size(0)

    # Sample random timesteps
    t = torch.randint(0, schedule.T, (batch_size,), device=x0.device)

    # Sample noise
    noise = torch.randn_like(x0)

    # Get noisy samples
    x_t = schedule.add_noise(x0, t, noise)

    # Predict noise
    noise_pred = model(x_t, t)

    # MSE loss
    loss = nn.functional.mse_loss(noise_pred, noise)

    return loss
```

### Generation: The Denoising Loop

Once trained, generation works by reversing the diffusion process. We start with pure noise and apply the learned denoising step repeatedly:

```python
@torch.no_grad()
def sample(model, schedule, shape, device='cpu'):
    """
    Generate samples using DDPM sampling
    """
    # Start from pure noise
    x = torch.randn(shape, device=device)

    for t in reversed(range(schedule.T)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        # Predict noise
        noise_pred = model(x, t_batch)

        # Compute mean
        alpha = schedule.alphas[t]
        alpha_bar = schedule.alpha_bars[t]
        beta = schedule.betas[t]

        mean = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_bar)) * noise_pred
        )

        # Add noise (except for t=0)
        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(beta)
            x = mean + sigma * noise
        else:
            x = mean

    return x
```

Notice that we add noise at each step except the last. This stochasticity is important for generating diverse samples. Without it, the same initial noise would always produce the same output.

### The Network Architecture: Time-Conditioned Denoising

The noise prediction network needs to know what timestep it is operating at. The same noisy input requires different processing depending on whether we are at step 900 (heavily noised, need aggressive denoising) or step 10 (lightly noised, need gentle cleanup).

The standard approach uses sinusoidal position embeddings (borrowed from transformers) to encode the timestep, and then injects this information into the network through various mechanisms:

```python
class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for timestep conditioning"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
```

For protein structures and sequences, U-Net style architectures with skip connections work well. The encoder progressively processes the input at multiple scales, the bottleneck captures global context, and the decoder reconstructs the denoised output while using skip connections to preserve fine details.

### Handling Discrete Data: The Protein Challenge

There is a subtle issue when applying diffusion to proteins. Diffusion models are designed for continuous data, Gaussian noise added to real-valued vectors. But protein sequences are discrete: each position is one of 20 amino acids (plus possibly a gap or special tokens).

Several approaches address this challenge:

**Continuous Relaxation**: Embed discrete tokens into a continuous space, apply diffusion there, and project back to discrete tokens at the end. This is simple and works reasonably well, though the projection step can introduce artifacts.

**Discrete Diffusion (D3PM)**: Define a discrete corruption process where tokens can transition to other tokens or to a special [MASK] token. Instead of Gaussian noise, we have transition matrices that gradually corrupt the sequence. The reverse process learns to predict the original tokens from the corrupted version.

$$q(x_t | x_{t-1}) = \text{Categorical}(x_t; Q_t x_{t-1})$$

where $Q_t$ is a transition matrix.

**Structure-Based Diffusion**: For protein structure generation (like RFDiffusion), we can work directly with 3D coordinates, which are naturally continuous. The sequence is either fixed, predicted from the structure, or jointly modeled.

---

## Part 3: VAEs vs Diffusion - Choosing Your Tool

Both VAEs and diffusion models are powerful generative approaches, but they have different strengths:

| Aspect | VAE | Diffusion |
|--------|-----|-----------|
| **Latent space** | Low-dimensional, structured | Same dimension as data |
| **Training** | Single forward pass | Multiple noise levels |
| **Sampling** | Single decoder pass | Many denoising steps (slow) |
| **Quality** | Good, but can be blurry | State-of-the-art |
| **Diversity** | High | Very high |
| **Control** | Via latent manipulation | Via guidance |
| **Interpretability** | Latent dimensions can be meaningful | Less interpretable |

VAEs shine when you want a structured latent space for interpolation, visualization, or property-guided design. You can train a VAE, project proteins into latent space, and see that similar proteins cluster together. You can move along latent dimensions and observe how protein properties change.

Diffusion models excel at sample quality. When you need the most realistic, physically plausible protein structures, diffusion is the current state-of-the-art. The iterative refinement process seems to capture fine details better than single-shot decoding.

For practical applications, the choice often depends on your computational budget. VAE sampling is fast (one decoder pass), while diffusion sampling requires hundreds or thousands of denoising steps. If you need to generate millions of candidates, VAEs might be more practical. If you need the best possible candidates and can afford slower generation, diffusion is the way to go.

---

## Part 4: Real-World Impact - From Theory to Therapeutics

### RFDiffusion: Designing Protein Structures from Scratch

RFDiffusion, developed by the Baker lab at the University of Washington, is arguably the most impactful application of diffusion models to protein science. It can generate entirely new protein backbone structures conditioned on various constraints: desired shape, binding targets, symmetry requirements, and more.

The model works in structure space, diffusing over the 3D coordinates of protein backbones. Starting from random coordinates (noise), it iteratively denoises to produce physically plausible structures. The key insight is that the denoising network is based on RoseTTAFold, a structure prediction model, so it has a deep understanding of what realistic protein structures look like.

RFDiffusion has been used to design:
- Novel protein binders for therapeutic targets
- Symmetric protein assemblies (rings, cages)
- Proteins with specific shapes and topologies
- Enzymes with new active site geometries

### EvoDiff: Generating Sequences with Evolutionary Awareness

While RFDiffusion works on structures, EvoDiff applies diffusion to protein sequences. Developed by researchers at Microsoft Research, it generates sequences using a discrete diffusion process that respects the evolutionary patterns learned from millions of natural sequences.

EvoDiff can generate:
- Unconditional protein sequences
- Sequences conditioned on protein families
- Inpainting: filling in missing regions of sequences

The generated sequences often fold into stable structures and show evolutionary-like patterns, suggesting the model has captured deep principles of protein sequence space.

### Conditional Generation: Designing with Intent

One of the most exciting directions is conditional generation, where we guide the generative model toward proteins with desired properties. For VAEs, this typically involves conditioning the decoder on property labels:

```python
class ConditionalProteinVAE(ProteinVAE):
    def __init__(self, seq_len, vocab_size=21, hidden_dim=256,
                 latent_dim=32, condition_dim=10):
        super().__init__(seq_len, vocab_size, hidden_dim, latent_dim)

        # Condition embedding
        self.cond_embed = nn.Linear(condition_dim, hidden_dim)

        # Modified decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * vocab_size)
        )

    def decode(self, z, condition):
        cond_emb = self.cond_embed(condition)
        z_cond = torch.cat([z, cond_emb], dim=-1)
        h = self.decoder(z_cond)
        return h.view(-1, self.seq_len, self.vocab_size)
```

For diffusion models, classifier guidance or classifier-free guidance allow steering generation toward desired properties without retraining the base model. The idea is to modify the denoising direction to increase the probability of the desired property:

```python
def guided_sample(model, schedule, shape, classifier, target_class,
                  guidance_scale=1.0, device='cpu'):
    """
    Classifier-guided diffusion sampling
    """
    x = torch.randn(shape, device=device)

    for t in reversed(range(schedule.T)):
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        with torch.enable_grad():
            x.requires_grad_(True)

            # Get classifier gradient toward target class
            logits = classifier(x, t_batch)
            log_prob = torch.log_softmax(logits, dim=-1)[:, target_class]
            grad = torch.autograd.grad(log_prob.sum(), x)[0]

        x = x.detach()

        # Guided noise prediction: shift toward high-probability regions
        noise_pred = model(x, t_batch)
        noise_pred = noise_pred - guidance_scale * torch.sqrt(
            1 - schedule.alpha_bars[t]) * grad

        # Standard DDPM update with guided noise
        # ... (same as before)

    return x
```

---

## Summary: Key Takeaways

We have covered a lot of ground in this lecture. Let us summarize the key insights:

**Variational Autoencoders** learn a probabilistic latent space that enables both compression and generation. The ELBO objective balances reconstruction quality with latent space regularity. The reparameterization trick makes it possible to train end-to-end through sampling operations.

**Diffusion Models** learn to reverse a gradual noising process. The forward process has a closed form, enabling efficient training at any timestep. The reverse process is learned by training a network to predict the noise. Generation requires many denoising steps but produces high-quality samples.

**For protein science**, both approaches have proven valuable. VAEs offer interpretable latent spaces for understanding protein families and guiding design. Diffusion models achieve state-of-the-art quality for structure and sequence generation.

**The practical impact** is real and growing. RFDiffusion has designed novel therapeutic proteins. EvoDiff generates evolutionarily-plausible sequences. These are not just academic exercises but tools that are actively being used in drug development and protein engineering.

As you continue in this field, remember that generative models are tools in your toolkit. The best tool depends on your specific application: What are you trying to generate? What constraints do you have? How much computation can you afford? Understanding the principles behind VAEs and diffusion will help you choose wisely and adapt as new methods emerge.

---

## References

1. Kingma, D.P. & Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR.
2. Ho, J., Jain, A. & Abbeel, P. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
3. Watson, J.L. et al. (2023). De novo design of protein structure and function with RFdiffusion. Nature.
4. Alamdari, S. et al. (2023). Protein generation with evolutionary diffusion. Nature Machine Intelligence.
5. Austin, J. et al. (2021). Structured Denoising Diffusion Models in Discrete State-Spaces. NeurIPS.
6. Higgins, I. et al. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. ICLR.

---

## Exercises

1. **Implement a protein VAE**: Train a VAE on a family of homologous sequences and visualize the latent space using t-SNE or UMAP. Do similar sequences cluster together?

2. **Explore the beta-VAE trade-off**: Vary $\beta$ from 0.1 to 10 and observe how it affects reconstruction quality versus latent space structure. At what $\beta$ values do you see disentanglement?

3. **Diffusion from scratch**: Implement DDPM training and sampling for 1D protein property data (e.g., per-residue conservation scores). Visualize the denoising process at different timesteps.

4. **Compare generation quality**: Generate sequences from trained VAE and diffusion models, evaluate with ESM perplexity. Which produces more "protein-like" sequences?

5. **Conditional generation**: Implement a conditional VAE that generates sequences based on desired secondary structure content. Can you generate sequences that are 50% helix, 30% sheet, 20% coil?
