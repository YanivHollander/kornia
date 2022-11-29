from typing import Tuple

import torch

from kornia.core import Tensor
from kornia.nerf.rays import calc_ray_t_vals


class VolumeRenderer(torch.nn.Module):
    r"""Volume renderer class. Implementation follows Ben Mildenhall et el. (2020) at
    https://arxiv.org/abs/2003.08934.

    Args:
        shift: Size of far-field layer: int
    """
    _huge = 1.0e10
    _eps = 1.0e-10

    def __init__(self, shift: int = 1) -> None:
        super().__init__()
        self._shift = shift

    def _render(self, alpha: Tensor, rgbs: Tensor) -> Tuple[Tensor, Tensor]:
        trans = torch.cumprod(1 - alpha + self._eps, dim=-1)  # (*, N)
        trans = torch.roll(trans, shifts=self._shift, dims=-1)  # (*, N)
        trans[..., : self._shift] = 1  # (*, N)

        weights = trans * alpha  # (*, N)

        rgbs_rendered = torch.sum(weights[..., None] * rgbs, dim=-2)  # (*, 3)

        return rgbs_rendered, weights

    def forward(self, rgbs: Tensor, densities: Tensor, points_3d: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class IrregularRenderer(VolumeRenderer):
    def __init__(self, shift: int = 1) -> None:
        super().__init__(shift)

    def forward(self, rgbs: Tensor, densities: Tensor, points_3d: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Renders 3D irregularly sampled points along rays.

        Args:
            rgbs: RGB values of points along rays :math:`(*, N, 3)`
            densities: Volume densities of points along rays :math:`(*, N)`
            points_3d: Ray points in world coordinates: math: `(*, N, 3)`

        Returns:
            Rendered RGB values for each ray :math:`(*, 3)`
        """
        # FIXME: Instead of calculating t values from 3D points, pass to 'forward' t values that were calculated by
        # the caller (NerfModel::forward)
        t_vals = calc_ray_t_vals(points_3d)
        deltas = t_vals[..., 1:] - t_vals[..., :-1]  # (*, N - 1)
        far = torch.empty(size=t_vals.shape[:-1], dtype=t_vals.dtype, device=t_vals.device).fill_(self._huge)
        deltas = torch.cat([deltas, far[..., None]], dim=-1)  # (*, N)

        alpha = 1 - torch.exp(-1.0 * densities * deltas)  # (*, N)

        return self._render(alpha, rgbs)


class RegularRenderer(VolumeRenderer):
    def __init__(self, shift: int = 1) -> None:
        super().__init__(shift)

    def forward(self, rgbs: Tensor, densities: Tensor, points_3d: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Renders 3D regularly sampled points along rays.

        Args:
            rgbs: RGB values of points along rays :math:`(*, N, 3)`
            densities: Volume densities of points along rays :math:`(*, N)`
            points_3d: Ray points in world coordinates: math: `(*, N, 3)`

        Returns:
            Rendered RGB values for each ray :math:`(*, 3)`
        """
        num_ray_points = points_3d.shape[-2]
        delta_3d = points_3d.reshape(-1, num_ray_points, 3)[0, 1, :] - points_3d.reshape(-1, num_ray_points, 3)[0, 0, :]
        delta = torch.linalg.norm(delta_3d, dim=-1)

        alpha = 1 - torch.exp(-1.0 * densities * delta)  # (*, N)

        return self._render(alpha, rgbs)
