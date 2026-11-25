"""Lightweight 3D Vision Mamba (Med-Mamba) Deep Feature Extractor in PyTorch.

Implements a Bidirectional Selective State Space Model (SSM) adapted for 3D
T1-weighted brain MRI volumes. Operates strictly in frozen inference mode
(torch.no_grad, eval) to extract compact, expressive anatomical spatial embeddings
without neural network training or fine-tuning from scratch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM3D(nn.Module):
    """Selective State Space Model (S6) core with 1D parameter recurrence.

    Discretizes continuous state space equations:
        h'(t) = A h(t) + B u(t)
        y(t)  = C h(t) + D u(t)
    via zero-order hold with input-dependent selection of Delta, B, and C.
    """

    def __init__(self, d_model: int, d_state: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Continuous A initialized with HiPPO-like transition spectrum
        A = torch.repeat_interleave(
            torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0),
            d_model,
            0,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

        # Input-dependent projections for Delta, B, C
        self.x_proj = nn.Linear(d_model, d_state * 2 + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        # Initialize dt_proj bias for numerical stability
        dt_init_std = 0.001
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Initialize Delta bias so softplus(Delta_bias) ~ 0.01-0.1
        dt = torch.exp(
            torch.rand(d_model) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def _selective_scan(
        self,
        u: torch.Tensor,
        A_log: torch.Tensor,
        B_param: torch.Tensor,
        C_param: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Sequential selective state space scan over sequence tokens."""
        b_sz, l_sz, d_sz = u.shape
        n_sz = self.d_state

        # Continuous to discrete discretization
        # delta_A: (B, L, D, N)
        delta_A = torch.exp(delta.unsqueeze(-1) * (-torch.exp(A_log)))
        # delta_B: (B, L, D, N)
        delta_B = delta.unsqueeze(-1) * B_param.unsqueeze(2)
        delta_B_u = delta_B * u.unsqueeze(-1)

        h = torch.zeros(b_sz, d_sz, n_sz, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(l_sz):
            h = delta_A[:, t] * h + delta_B_u[:, t]
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(dim=-1)  # (B, D)
            ys.append(y_t)
        return torch.stack(ys, dim=1)  # (B, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bidirectional SSM scan: captures both forward and reverse 3D spatial contexts."""
        b_sz, l_sz, d_sz = x.shape
        x_proj = self.x_proj(x)
        delta_raw, B_param, C_param = torch.split(
            x_proj, [d_sz, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_raw))

        # 1. Forward scan (Spatial order 0 -> L)
        y_fwd = self._selective_scan(x, self.A_log, B_param, C_param, delta)

        # 2. Backward scan (Spatial order L -> 0)
        x_rev = torch.flip(x, dims=[1])
        delta_rev = torch.flip(delta, dims=[1])
        B_rev = torch.flip(B_param, dims=[1])
        C_rev = torch.flip(C_param, dims=[1])
        y_bwd = torch.flip(
            self._selective_scan(x_rev, self.A_log, B_rev, C_rev, delta_rev),
            dims=[1],
        )

        # Merge bidirectional representations with skip connection D
        return y_fwd + y_bwd + x * self.D


class VisionMambaBlock3D(nn.Module):
    """3D Vision Mamba block combining spatial convolution and selective SSM."""

    def __init__(self, d_model: int = 48, d_state: int = 16, expand: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner
        )
        self.act = nn.SiLU()
        self.ssm = SelectiveSSM3D(d_inner, d_state=d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        x_inner, z = torch.chunk(xz, 2, dim=-1)

        # Local sequence context via depthwise 1D conv
        x_conv = self.act(
            self.conv1d(x_inner.transpose(1, 2)).transpose(1, 2)
        )
        # Global non-local context via Bidirectional SSM
        y = self.ssm(x_conv)
        # Gating
        y = y * self.act(z)
        out = self.out_proj(y) + residual
        return out


class MedMamba3D(nn.Module):
    """3D Vision Mamba (Med-Mamba) Deep Feature Extractor for Brain MRI.

    Converts a standardized 3D brain MRI volume into a compact deep anatomical
    embedding vector (e.g. 24 dimensions) using 3D patch tokenization and
    bidirectional selective state space modeling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        patch_size: int = 16,
        d_model: int = 48,
        d_state: int = 16,
        depth: int = 2,
        embedding_dim: int = 24,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size

        # 3D Patch Embedding: non-overlapping 16x16x16 patches
        self.patch_embed = nn.Conv3d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        self.blocks = nn.ModuleList(
            [
                VisionMambaBlock3D(d_model=d_model, d_state=d_state)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Freeze model parameters
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects x of shape (B, 1, D, H, W)."""
        feat = self.patch_embed(x)  # (B, d_model, D', H', W')
        b_sz, c_sz, d_p, h_p, w_p = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, L, C)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)  # (B, d_model)
        embedding = self.head(pooled)  # (B, embedding_dim)
        return embedding


_GLOBAL_MAMBA_MODEL: Optional[MedMamba3D] = None


def get_frozen_mamba_model(embedding_dim: int = 24, device: str = "cpu") -> MedMamba3D:
    """Retrieve or instantiate the deterministic singleton frozen MedMamba3D model."""
    global _GLOBAL_MAMBA_MODEL
    if _GLOBAL_MAMBA_MODEL is None or _GLOBAL_MAMBA_MODEL.embedding_dim != embedding_dim:
        _GLOBAL_MAMBA_MODEL = MedMamba3D(embedding_dim=embedding_dim, seed=42)
        _GLOBAL_MAMBA_MODEL.to(device)
        _GLOBAL_MAMBA_MODEL.eval()
    return _GLOBAL_MAMBA_MODEL


def preprocess_volume_for_mamba(
    volume: np.ndarray,
    target_shape: tuple[int, int, int] = (96, 96, 96),
) -> torch.Tensor:
    """Prepare a 3D numpy brain volume for MedMamba3D.

    Normalizes intensity to zero-mean unit-variance on non-zero tissue,
    crops or resizes to target shape (96, 96, 96), and returns a 5D PyTorch
    tensor of shape (1, 1, 96, 96, 96).
    """
    data = volume.astype(np.float32)

    # 1. Non-zero z-score normalization
    nonzero_mask = data > 0
    if np.any(nonzero_mask):
        mean_v = float(np.mean(data[nonzero_mask]))
        std_v = float(np.std(data[nonzero_mask]))
        if std_v > 1e-6:
            data[nonzero_mask] = (data[nonzero_mask] - mean_v) / std_v
            data[~nonzero_mask] = 0.0

    # 2. Convert to PyTorch Tensor: (1, 1, D, H, W)
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)

    # 3. Trilinear interpolation to uniform dimensions (96, 96, 96)
    if tensor.shape[2:] != target_shape:
        tensor = F.interpolate(
            tensor,
            size=target_shape,
            mode="trilinear",
            align_corners=False,
        )

    return tensor.float()


def extract_mamba_features(
    input_source: Union[str, Path, np.ndarray, nib.Nifti1Image],
    embedding_dim: int = 24,
    device: str = "cpu",
    model: Optional[MedMamba3D] = None,
) -> dict[str, float]:
    """Extract compact 3D Vision Mamba anatomical embedding vector from an MRI scan.

    Returns a dictionary mapping feature names 'mamba_feat_00', 'mamba_feat_01', ...
    to their numerical values.
    """
    if isinstance(input_source, (str, Path)):
        img = nib.load(str(input_source))
        volume = img.get_fdata(dtype=np.float32)
    elif isinstance(input_source, nib.Nifti1Image):
        volume = input_source.get_fdata(dtype=np.float32)
    elif isinstance(input_source, np.ndarray):
        volume = input_source
    else:
        raise TypeError(f"Unsupported input type for extract_mamba_features: {type(input_source)}")

    if model is None:
        model = get_frozen_mamba_model(embedding_dim=embedding_dim, device=device)

    tensor_input = preprocess_volume_for_mamba(volume).to(device)

    with torch.no_grad():
        emb_tensor = model(tensor_input)  # Shape (1, embedding_dim)
        emb_vec = emb_tensor.squeeze(0).cpu().numpy().tolist()

    feature_dict = {
        f"mamba_feat_{i:02d}": float(val)
        for i, val in enumerate(emb_vec)
    }
    return feature_dict
