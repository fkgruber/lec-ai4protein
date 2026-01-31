"""
Mini AlphaFold Implementation

A simplified educational implementation of AlphaFold2's core components.
This version omits MSA processing and uses single sequence input.

For educational purposes only - not for production use.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(1)]


class PairwisePositionalEncoding(nn.Module):
    """Relative positional encoding for pair representation."""

    def __init__(self, d_pair: int, max_dist: int = 32):
        super().__init__()
        self.max_dist = max_dist
        self.embedding = nn.Embedding(2 * max_dist + 1, d_pair)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Generate relative position encoding."""
        idx = torch.arange(seq_len, device=self.embedding.weight.device)
        rel_pos = idx[:, None] - idx[None, :]
        rel_pos = torch.clamp(rel_pos + self.max_dist, 0, 2 * self.max_dist)
        return self.embedding(rel_pos)


class SimplifiedEvoformerBlock(nn.Module):
    """
    Simplified Evoformer block for single sequence input.

    Processes single representation and pair representation through:
    1. Self-attention on sequence with pair bias
    2. Pair representation update via triangular attention
    3. Outer product to update pairs from sequence
    """

    def __init__(self, d_single: int = 256, d_pair: int = 128, n_heads: int = 8):
        super().__init__()
        self.d_single = d_single
        self.d_pair = d_pair
        self.n_heads = n_heads
        self.head_dim = d_single // n_heads

        # Single representation attention with pair bias
        self.norm_single = nn.LayerNorm(d_single)
        self.norm_pair_bias = nn.LayerNorm(d_pair)
        self.to_qkv = nn.Linear(d_single, 3 * d_single)
        self.pair_bias = nn.Linear(d_pair, n_heads)
        self.out_proj = nn.Linear(d_single, d_single)

        # Single representation transition
        self.norm_ffn = nn.LayerNorm(d_single)
        self.ffn = nn.Sequential(
            nn.Linear(d_single, d_single * 4),
            nn.GELU(),
            nn.Linear(d_single * 4, d_single)
        )

        # Triangular multiplicative update (simplified)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.tri_left = nn.Linear(d_pair, d_pair)
        self.tri_right = nn.Linear(d_pair, d_pair)
        self.tri_out = nn.Linear(d_pair, d_pair)

        # Outer product mean (single -> pair)
        self.norm_opm = nn.LayerNorm(d_single)
        self.opm_left = nn.Linear(d_single, 32)
        self.opm_right = nn.Linear(d_single, 32)
        self.opm_out = nn.Linear(32 * 32, d_pair)

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            single: [B, L, d_single] single representation
            pair: [B, L, L, d_pair] pair representation

        Returns:
            Updated single and pair representations
        """
        B, L, _ = single.shape

        # 1. Self-attention with pair bias
        s = self.norm_single(single)
        qkv = self.to_qkv(s).chunk(3, dim=-1)
        q, k, v = [x.view(B, L, self.n_heads, self.head_dim).transpose(1, 2) for x in qkv]

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Add pair bias
        bias = self.pair_bias(self.norm_pair_bias(pair))  # [B, L, L, n_heads]
        attn = attn + bias.permute(0, 3, 1, 2)

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_single)
        single = single + self.out_proj(out)

        # 2. FFN transition
        single = single + self.ffn(self.norm_ffn(single))

        # 3. Triangular multiplicative update (simplified outgoing)
        p = self.norm_pair(pair)
        left = self.tri_left(p)  # [B, L, L, d_pair]
        right = self.tri_right(p)
        # Simplified: just use element-wise product along one axis
        tri_out = torch.einsum('bikc,bjkc->bijc', left, right) / L
        pair = pair + self.tri_out(tri_out)

        # 4. Outer product mean
        s = self.norm_opm(single)
        left = self.opm_left(s)  # [B, L, 32]
        right = self.opm_right(s)  # [B, L, 32]
        outer = torch.einsum('bic,bjd->bijcd', left, right)
        outer = outer.view(B, L, L, -1)
        pair = pair + self.opm_out(outer)

        return single, pair


class InvariantPointAttention(nn.Module):
    """
    Simplified Invariant Point Attention.

    This is a basic version that operates on:
    - Single representation features
    - Pairwise features
    - 3D coordinates
    """

    def __init__(self, d_single: int = 256, d_pair: int = 128, n_heads: int = 4):
        super().__init__()
        self.d_single = d_single
        self.n_heads = n_heads
        self.head_dim = d_single // n_heads

        # Standard attention
        self.norm_single = nn.LayerNorm(d_single)
        self.to_q = nn.Linear(d_single, d_single)
        self.to_k = nn.Linear(d_single, d_single)
        self.to_v = nn.Linear(d_single, d_single)

        # Pair bias
        self.pair_bias = nn.Linear(d_pair, n_heads)

        # Point attention (3D coordinates)
        self.n_points = 4
        self.to_q_pts = nn.Linear(d_single, n_heads * self.n_points * 3)
        self.to_k_pts = nn.Linear(d_single, n_heads * self.n_points * 3)
        self.to_v_pts = nn.Linear(d_single, n_heads * self.n_points * 3)

        # Output
        self.out_proj = nn.Linear(d_single + n_heads * self.n_points * 3, d_single)

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            single: [B, L, d_single]
            pair: [B, L, L, d_pair]
            coords: [B, L, 3] CA coordinates

        Returns:
            Updated single representation
        """
        B, L, _ = single.shape

        s = self.norm_single(single)

        # Standard attention components
        q = self.to_q(s).view(B, L, self.n_heads, self.head_dim)
        k = self.to_k(s).view(B, L, self.n_heads, self.head_dim)
        v = self.to_v(s).view(B, L, self.n_heads, self.head_dim)

        # Attention scores from features
        attn = torch.einsum('bihd,bjhd->bhij', q, k) / math.sqrt(self.head_dim)

        # Add pair bias
        bias = self.pair_bias(pair)  # [B, L, L, n_heads]
        attn = attn + bias.permute(0, 3, 1, 2)

        # Point-based attention
        q_pts = self.to_q_pts(s).view(B, L, self.n_heads, self.n_points, 3)
        k_pts = self.to_k_pts(s).view(B, L, self.n_heads, self.n_points, 3)
        v_pts = self.to_v_pts(s).view(B, L, self.n_heads, self.n_points, 3)

        # Add coordinate-based offset
        coords_expanded = coords[:, :, None, None, :]  # [B, L, 1, 1, 3]
        q_pts = q_pts + coords_expanded
        k_pts = k_pts + coords_expanded

        # Compute point-based attention (distance-based)
        # [B, L, H, P, 3] -> compare all pairs
        pt_diff = q_pts[:, :, None, :, :, :] - k_pts[:, None, :, :, :, :]  # [B, L, L, H, P, 3]
        pt_dist = (pt_diff ** 2).sum(dim=-1).sum(dim=-1)  # [B, L, L, H]
        attn = attn - 0.5 * pt_dist.permute(0, 3, 1, 2)  # Closer = higher attention

        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out_feat = torch.einsum('bhij,bjhd->bihd', attn, v)
        out_feat = out_feat.reshape(B, L, self.d_single)

        # Apply attention to point values
        out_pts = torch.einsum('bhij,bjhpc->bihpc', attn, v_pts)
        out_pts = out_pts.reshape(B, L, -1)

        # Combine and project
        out = torch.cat([out_feat, out_pts], dim=-1)
        return single + self.out_proj(out)


class StructureModule(nn.Module):
    """
    Structure Module that converts representations to 3D coordinates.

    Uses iterative refinement with IPA.
    """

    def __init__(
        self,
        d_single: int = 256,
        d_pair: int = 128,
        n_layers: int = 4
    ):
        super().__init__()
        self.n_layers = n_layers

        # Initial coordinate prediction
        self.init_coords = nn.Linear(d_single, 3)

        # IPA layers
        self.ipa_layers = nn.ModuleList([
            InvariantPointAttention(d_single, d_pair)
            for _ in range(n_layers)
        ])

        # Coordinate updates
        self.coord_updates = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_single),
                nn.Linear(d_single, 3)
            )
            for _ in range(n_layers)
        ])

        # Transitions
        self.transitions = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_single),
                nn.Linear(d_single, d_single * 4),
                nn.GELU(),
                nn.Linear(d_single * 4, d_single)
            )
            for _ in range(n_layers)
        ])

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            single: [B, L, d_single]
            pair: [B, L, L, d_pair]

        Returns:
            coords: [B, L, 3] predicted CA coordinates
            single: [B, L, d_single] final single representation
        """
        # Initialize coordinates
        coords = self.init_coords(single)

        # Iterative refinement
        for ipa, update, transition in zip(
            self.ipa_layers, self.coord_updates, self.transitions
        ):
            # IPA attention
            single = ipa(single, pair, coords)

            # Transition
            single = single + transition(single)

            # Update coordinates
            coords = coords + update(single)

        return coords, single


class MiniAlphaFold(nn.Module):
    """
    Mini AlphaFold: A simplified educational implementation.

    Key differences from full AlphaFold2:
    - Single sequence input (no MSA)
    - Simplified Evoformer (no column attention)
    - Simplified IPA (no quaternion frames)
    - No recycling

    Architecture:
    1. Sequence embedding
    2. Pair representation initialization
    3. Simplified Evoformer blocks
    4. Structure module with IPA
    """

    def __init__(
        self,
        vocab_size: int = 21,
        d_single: int = 256,
        d_pair: int = 128,
        n_evoformer_blocks: int = 4,
        n_structure_layers: int = 4,
        n_heads: int = 8,
        max_len: int = 512
    ):
        super().__init__()
        self.d_single = d_single
        self.d_pair = d_pair

        # Sequence embedding
        self.aa_embed = nn.Embedding(vocab_size, d_single)
        self.pos_embed = SinusoidalPositionalEncoding(d_single, max_len)

        # Pair representation initialization
        self.left_proj = nn.Linear(d_single, d_pair)
        self.right_proj = nn.Linear(d_single, d_pair)
        self.rel_pos = PairwisePositionalEncoding(d_pair)

        # Evoformer blocks
        self.evoformer_blocks = nn.ModuleList([
            SimplifiedEvoformerBlock(d_single, d_pair, n_heads)
            for _ in range(n_evoformer_blocks)
        ])

        # Structure module
        self.structure_module = StructureModule(
            d_single, d_pair, n_structure_layers
        )

        # pLDDT prediction head
        self.plddt_head = nn.Sequential(
            nn.LayerNorm(d_single),
            nn.Linear(d_single, d_single),
            nn.GELU(),
            nn.Linear(d_single, 50)  # 50 pLDDT bins
        )

    def forward(
        self,
        seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass.

        Args:
            seq: [B, L] amino acid indices (0-20)
            mask: [B, L] attention mask (optional)

        Returns:
            Dictionary with:
            - coords: [B, L, 3] predicted CA coordinates
            - plddt: [B, L] predicted confidence scores
            - single: [B, L, d_single] final single representation
            - pair: [B, L, L, d_pair] final pair representation
        """
        B, L = seq.shape

        # Embed sequence
        single = self.aa_embed(seq)  # [B, L, d_single]
        single = self.pos_embed(single)

        # Initialize pair representation
        left = self.left_proj(single)   # [B, L, d_pair]
        right = self.right_proj(single)  # [B, L, d_pair]
        pair = left[:, :, None, :] + right[:, None, :, :]  # [B, L, L, d_pair]

        # Add relative position encoding
        pair = pair + self.rel_pos(L)[None, :, :, :]

        # Evoformer
        for block in self.evoformer_blocks:
            single, pair = block(single, pair)

        # Structure module
        coords, single = self.structure_module(single, pair)

        # pLDDT prediction
        plddt_logits = self.plddt_head(single)
        plddt = torch.softmax(plddt_logits, dim=-1)
        # Convert to 0-100 scale
        bins = torch.linspace(0, 1, 50, device=plddt.device)
        plddt_score = (plddt * bins).sum(dim=-1) * 100

        return {
            'coords': coords,
            'plddt': plddt_score,
            'single': single,
            'pair': pair
        }


def fape_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    clamp_distance: float = 10.0
) -> torch.Tensor:
    """
    Simplified Frame Aligned Point Error loss.

    This version uses each residue's coordinate as frame origin
    (simplified - no rotation).

    Args:
        pred_coords: [B, L, 3] predicted coordinates
        true_coords: [B, L, 3] ground truth coordinates
        clamp_distance: maximum distance for clamping

    Returns:
        FAPE loss (scalar)
    """
    B, L, _ = pred_coords.shape

    # Compute local coordinates (relative to each residue)
    # pred_local[b, i, j] = pred_coords[b, j] - pred_coords[b, i]
    pred_local = pred_coords[:, None, :, :] - pred_coords[:, :, None, :]  # [B, L, L, 3]
    true_local = true_coords[:, None, :, :] - true_coords[:, :, None, :]

    # Compute error
    error = torch.sqrt(((pred_local - true_local) ** 2).sum(dim=-1) + 1e-8)

    # Clamp
    error = torch.clamp(error, max=clamp_distance)

    return error.mean()


def distogram_loss(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    n_bins: int = 64,
    min_dist: float = 2.0,
    max_dist: float = 22.0
) -> torch.Tensor:
    """
    Distogram loss: predict pairwise distance distributions.

    Args:
        pred_coords: [B, L, 3] predicted coordinates
        true_coords: [B, L, 3] ground truth coordinates

    Returns:
        Distogram cross-entropy loss
    """
    # Compute true distances
    true_dist = torch.sqrt(
        ((true_coords[:, :, None, :] - true_coords[:, None, :, :]) ** 2).sum(dim=-1)
        + 1e-8
    )  # [B, L, L]

    # Bin distances
    bins = torch.linspace(min_dist, max_dist, n_bins, device=true_dist.device)
    true_bins = torch.bucketize(true_dist, bins)  # [B, L, L]

    # Compute predicted distances
    pred_dist = torch.sqrt(
        ((pred_coords[:, :, None, :] - pred_coords[:, None, :, :]) ** 2).sum(dim=-1)
        + 1e-8
    )

    # Simple MSE on distances (could be extended to predict distribution)
    return F.mse_loss(pred_dist, true_dist)


if __name__ == "__main__":
    # Test the model
    model = MiniAlphaFold()

    # Create dummy input
    batch_size = 2
    seq_len = 50
    seq = torch.randint(0, 21, (batch_size, seq_len))

    # Forward pass
    output = model(seq)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Predicted coords shape: {output['coords'].shape}")
    print(f"pLDDT shape: {output['plddt'].shape}")
    print(f"pLDDT range: [{output['plddt'].min():.1f}, {output['plddt'].max():.1f}]")

    # Test loss
    true_coords = torch.randn(batch_size, seq_len, 3) * 10
    loss = fape_loss(output['coords'], true_coords)
    print(f"FAPE loss: {loss.item():.4f}")
