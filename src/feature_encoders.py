import torch
from torch import nn
import numpy as np
from e3nn import o3


def xyz_to_angles(xyz):
    """Convert 3D vectors to spherical angles alpha and beta."""
    xyz = torch.nn.functional.normalize(xyz, p=2.0, dim=-1)
    xyz = xyz.clamp(-1, 1)
    beta = torch.acos(xyz[..., 1])
    alpha = torch.atan2(xyz[..., 0], xyz[..., 2])
    return alpha, beta


class BvecAngleEncoder(nn.Module):
    """
    Encodes selected vector components as spherical angles and appends them to the remaining coordinates.
    """
    def __init__(self, coord_size, vec_idx=(0, 1, 2)):
        super().__init__()
        self.coord_size = coord_size
        self.vec_idx = vec_idx

    @property
    def in_size(self):
        return self.coord_size

    @property
    def out_size(self):
        # Remove 3 vector components, add 2 angles
        return self.coord_size - len(self.vec_idx) + 2

    def forward(self, coords):
        # Extract vector part and compute angles
        vec = coords[..., list(self.vec_idx)]
        alpha, beta = xyz_to_angles(vec)
        # Gather remaining coordinates
        mask = torch.ones(coords.shape[-1], dtype=torch.bool, device=coords.device)
        mask[list(self.vec_idx)] = False
        rest = coords[..., mask]
        # Concatenate remaining coords with angles
        return torch.cat([rest, alpha.unsqueeze(-1), beta.unsqueeze(-1)], dim=-1)


class SphericalPosEncoder(nn.Module):
    """
    Spherical positional encoder combining random Fourier features and spherical harmonics.
    """
    def __init__(self, coord_size, l_max, freq_num, freq_scale=1.0, antipodal=True):
        super().__init__()
        self.coord_size = coord_size
        self.l_max = l_max
        self.freq_num = freq_num
        self.freq_scale = freq_scale
        self.antipodal = antipodal

        # Random Gaussian matrix for Fourier features
        B = torch.randn(coord_size, freq_num) * freq_scale
        self.register_buffer("B", B)
        self.register_buffer("B_pi", 2. * np.pi * B)

        # Spherical harmonics degree list
        if antipodal:
            self.l_values = list(range(0, l_max + 1, 2))
        else:
            self.l_values = list(range(l_max + 1))

    @property
    def out_size(self):
        # Fourier features + spherical harmonics
        return 2 * self.freq_num + sum([2 * l + 1 for l in self.l_values])

    @property
    def in_size(self):
        # spatial coords + 2 angles
        return self.coord_size + 2

    def get_extra_state(self):
        # Save state for reproducibility/checkpointing
        return {"B_pi": self.B_pi}

    def set_extra_state(self, state):
        self.B_pi.copy_(state["B_pi"])

    def forward(self, coords):
        # coords: [..., coord_size + 2] (spatial + 2 angles)
        assert coords.shape[-1] == self.coord_size + 2

        # Split input
        spatial = coords[..., :self.coord_size]
        alpha = coords[..., self.coord_size]
        beta = coords[..., self.coord_size + 1]

        # Fourier features
        x_proj = spatial @ self.B_pi
        fourier_feats = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

        # Spherical harmonics
        sph_feats = o3.spherical_harmonics_alpha_beta(self.l_values, alpha, beta)

        # Concatenate
        return torch.cat([fourier_feats, sph_feats], dim=-1)
