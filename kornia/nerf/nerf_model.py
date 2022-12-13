from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from kornia.core import Tensor
from kornia.nerf.positional_encoder import PositionalEncoder
from kornia.nerf.rays import sample_lengths, sample_ray_points
from kornia.nerf.renderer import IrregularRenderer, RegularRenderer


class MLP(nn.Module):
    r"""Class to represent a multi-layer perceptron. The MLP represents a deep NN of fully connected layers. The
    network is build of user defined sub-units, each with a user defined number of layers.  Skip connections span
    between the sub-units. The model follows: Ben Mildenhall et el. (2020) at https://arxiv.org/abs/2003.08934.

    Args:
        num_dims: Number of input dimensions (channels)
        num_units: Number of sub-units: int
        num_unit_layers: Number of fully connected layers in each sub-unit
        num_hidden: Layer hidden dimensions: int
    """

    def __init__(self, num_dims, num_units: int = 2, num_unit_layers: int = 4, num_hidden: int = 128):
        super().__init__()
        self._num_unit_layers = num_unit_layers
        layers = []
        for i in range(num_units):
            num_unit_inp_dims = num_dims if i == 0 else num_hidden + num_dims
            for j in range(num_unit_layers):
                num_layer_inp_dims = num_unit_inp_dims if j == 0 else num_hidden
                layer = nn.Linear(num_layer_inp_dims, num_hidden)
                layers.append(nn.Sequential(layer, nn.ReLU()))
        self._mlp = nn.ModuleList(layers)

    def forward(self, inp: Tensor) -> Tensor:
        out = inp
        inp_skip = inp
        for i, layer in enumerate(self._mlp):
            if i > 0 and i % self._num_unit_layers == 0:
                out = torch.cat((out, inp_skip), dim=-1)
            out = layer(out)
        return out


class NerfModel(nn.Module):
    r"""Class to represent NeRF model.

    Args:
        num_ray_points: Number of points to sample along rays: int
        irregular_ray_sampling: Whether to sample ray points irregularly: bool
        num_pos_freqs: Number of frequencies for positional encoding: int
        num_dir_freqs: Number of frequencies for directional encoding: int
        num_units: Number of sub-units: int
        num_unit_layers: Number of fully connected layers in each sub-unit
        num_hidden: Layer hidden dimensions: int
        log_space_encoding: Whether to apply log spacing for encoding: bool
        hierarchical_sampling: Apply two hierarchical sampling passes: bool
    """

    def __init__(
        self,
        num_ray_points: int,
        irregular_ray_sampling: bool = True,
        num_pos_freqs: int = 10,
        num_dir_freqs: int = 4,
        num_units: int = 2,
        num_unit_layers: int = 4,
        num_hidden: int = 128,
        log_space_encoding=True,
        hierarchical_sampling=False,
    ):
        super().__init__()
        self._num_ray_points = num_ray_points
        self._irregular_ray_sampling = irregular_ray_sampling
        self._renderer = IrregularRenderer() if self._irregular_ray_sampling else RegularRenderer()

        self._pos_encoder = PositionalEncoder(3, num_pos_freqs, log_space=log_space_encoding)
        self._dir_encoder = PositionalEncoder(3, num_dir_freqs, log_space=log_space_encoding)
        self._mlp = MLP(self._pos_encoder.num_encoded_dims, num_units, num_unit_layers, num_hidden)
        self._fc1 = nn.Linear(num_hidden, num_hidden)
        self._fc2 = nn.Sequential(
            nn.Linear(num_hidden + self._dir_encoder.num_encoded_dims, num_hidden // 2), nn.ReLU()
        )

        self._sigma = nn.Linear(num_hidden, 1)
        torch.nn.init.xavier_uniform_(self._sigma.weight.data)
        self._density_noise = 0.01

        self._rgb = nn.Sequential(nn.Linear(num_hidden // 2, 3), nn.Sigmoid())

        self._hierarchical_sampling = hierarchical_sampling

    def __forward_for_lengths(self, origins: Tensor, directions: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        points_3d = sample_ray_points(origins, directions, lengths)

        # Encode positions & directions
        points_3d_encoded = self._pos_encoder(points_3d)
        directions_encoded = self._dir_encoder(F.normalize(directions, dim=-1))

        # Map positional encodings to latent features (MLP with skip connections)
        y = self._mlp(points_3d_encoded)
        y = self._fc1(y)

        # Calculate ray point density values
        densities_ray_points = self._sigma(y).squeeze(-1)
        if self._density_noise > 0:
            densities_ray_points += self._density_noise * torch.normal(
                mean=torch.zeros_like(densities_ray_points), std=1.0
            )
        densities_ray_points = torch.nn.Softplus()(densities_ray_points - 1.0)

        # Calculate ray point rgb values
        y = torch.cat((y, directions_encoded[..., None, :].expand(-1, self._num_ray_points, -1)), dim=-1)
        y = self._fc2(y)
        rgbs_ray_points = self._rgb(y)

        # Rendering rgbs and densities along rays
        rgbs, weights = self._renderer(rgbs_ray_points, densities_ray_points, points_3d)

        # Return pixel point rendered rgb
        return rgbs, weights

    def forward(self, origins: Tensor, directions: Tensor) -> Tensor:

        # Sample xyz for ray parameters
        sampling_type = 'regular' if not self._irregular_ray_sampling or self._hierarchical_sampling else 'irregular'
        batch_size = origins.shape[0]
        lengths = sample_lengths(
            batch_size, self._num_ray_points, device=origins.device, dtype=origins.dtype, sampling_type=sampling_type
        )
        rgbs, weights = self.__forward_for_lengths(origins, directions, lengths)

        # Hierarchical sampling: with weights resampling lengths and run model again
        if self.training and self._hierarchical_sampling:

            # Resample lengths based on weights from the first pass
            weights = 0.5 * (weights[..., 1:] + weights[..., :-1])
            lengths = sample_lengths(
                batch_size,
                self._num_ray_points,
                device=origins.device,
                dtype=origins.dtype,
                sampling_type='piecewise_constant',
                bins=lengths,
                weights=weights,
            )
            rgbs, _ = self.__forward_for_lengths(origins, directions, lengths)

        return rgbs
