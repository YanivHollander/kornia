from datetime import datetime
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

from kornia.core import Module, Tensor, zeros
from kornia.geometry.camera import PinholeCamera
from kornia.metrics import psnr
from kornia.nerf.core import Images, ImageTensors
from kornia.nerf.data_utils import RandomBatchRayDataset, RayDataset
from kornia.nerf.nerf_model import NerfModel
from kornia.utils._compat import torch_inference_mode


class NerfParams:
    def __init__(
        self,
        min_depth: float = 0.0,
        max_depth: float = 1.0,
        batch_size: int = 4096,
        lr: float = 5.0e-4,
        num_ray_points: int = 64,
        irregular_ray_sampling: bool = False,
        hierarchical_sampling: bool = True,
        log_space_encoding: bool = True,
        num_pos_freqs: int = 10,
        num_dir_freqs: int = 4,
        num_units: int = 2,
        num_unit_layers: int = 4,
        num_hidden: int = 128,
    ) -> None:
        self._min_depth = min_depth
        self._max_depth = max_depth

        self._batch_size = batch_size
        self._lr = lr

        self._num_ray_points = num_ray_points
        self._irregular_ray_sampling = irregular_ray_sampling
        self._hierarchical_sampling = hierarchical_sampling
        self._log_space_encoding = log_space_encoding

        # NeRF model
        self._num_pos_freqs = num_pos_freqs
        self._num_dir_freqs = num_dir_freqs
        self._num_units = num_units
        self._num_unit_layers = num_unit_layers
        self._num_hidden = num_hidden


class NerfSolver:
    r"""NeRF solver class.

    Args:
        device: device for class tensors: Union[str, Device]
        dtype: type for all floating point calculations: torch.dtype
    """

    def __init__(self, device: torch.device, dtype: torch.dtype, params: NerfParams = NerfParams()) -> None:
        self._cameras: Optional[PinholeCamera] = None
        self._imgs: Optional[Images] = None

        self._device = device
        self._dtype = dtype

        self.set_nerf_params(params)

    def set_nerf_params(self, params: NerfParams) -> None:
        self._params = params
        self.__init_model()

    def __init_model(self) -> None:
        self._nerf_model = NerfModel(
            self._params._num_ray_points,
            irregular_ray_sampling=self._params._irregular_ray_sampling,
            num_pos_freqs=self._params._num_pos_freqs,
            num_dir_freqs=self._params._num_dir_freqs,
            num_units=self._params._num_units,
            num_unit_layers=self._params._num_unit_layers,
            num_hidden=self._params._num_hidden,
            log_space_encoding=self._params._log_space_encoding,
            hierarchical_sampling=self._params._hierarchical_sampling,
        )
        self._nerf_model.to(device=self._device, dtype=self._dtype)
        self._opt_nerf = optim.Adam(self._nerf_model.parameters(), lr=self._params._lr)

    def set_cameras_and_images_for_training(self, cameras: PinholeCamera, imgs: Images) -> None:
        self._cameras = cameras
        self._imgs = imgs

    @property
    def nerf_model(self) -> Module:
        return self._nerf_model

    def __step(self, origins: Tensor, directions: Tensor, rgbs: Tensor) -> float:
        rgbs_model = self._nerf_model(origins, directions)
        loss = F.mse_loss(rgbs_model, rgbs)

        step_psnr = psnr(rgbs_model, rgbs, 1.0)  # FIXME: This is a bit wasteful - calculating tensor diff 2nd time

        self._opt_nerf.zero_grad()
        loss.backward()
        self._opt_nerf.step()
        return float(step_psnr)

    def run(self, num_iters: int) -> None:
        def check_camera_image_consistency(cameras: PinholeCamera, imgs: Images):
            if cameras is None:
                raise TypeError('Invalid camera object')
            if imgs is None:
                raise TypeError('Invalid image list object')
            if cameras.batch_size != len(imgs):
                raise ValueError('Number of cameras must match number of input images')

            if not all(img.shape[0] == 3 for img in imgs):
                raise ValueError('All images must have three RGB channels')
            if not all(height == img.shape[1] for height, img in zip(cameras.height.tolist(), imgs)):
                raise ValueError('All image heights must match camera heights')
            if not all(width == img.shape[2] for width, img in zip(cameras.width.tolist(), imgs)):
                raise ValueError('All image widths must match camera widths')

        check_camera_image_consistency(self._cameras, self._imgs)
        self._nerf_model.train()
        rand_batch_ray_dataset = RandomBatchRayDataset(
            self._cameras,
            batch_size=self._params._batch_size,
            min_depth=self._params._min_depth,
            max_depth=self._params._max_depth,
            device=self._device,
            dtype=self._dtype,
            imgs=self._imgs,
        )
        for i_iter in range(num_iters):
            origins, directions, rgbs = rand_batch_ray_dataset.get_batch()
            iter_psnr = self.__step(origins, directions, rgbs)

            if i_iter % 10 == 0:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f'Iteration: {i_iter}: iter_psnr = {iter_psnr}; time: {current_time}')

    def render_views(self, cameras: PinholeCamera) -> ImageTensors:
        r"""Renders a novel synthesis view of a trained NeRF model for given cameras.

        Args:
            cameras: cameras for image renderings: PinholeCamera

        Returns:
            Rendered images: ImageTensors (List[(H, W, C)]).
        """
        self._nerf_model.eval()
        ray_dataset = RayDataset(
            cameras, self._params._min_depth, self._params._max_depth, False, device=self._device, dtype=self._dtype
        )
        ray_dataset.init_ray_dataset()
        idx0 = 0
        imgs: ImageTensors = []
        batch_size = 10  # FIXME: Consider exposing this value to the user
        for height, width in zip(cameras.height.int().tolist(), cameras.width.int().tolist()):
            bsz = batch_size if batch_size != -1 else height * width
            img = zeros((height * width, 3), dtype=torch.uint8)
            idx0_camera = idx0
            for idx0 in range(idx0, idx0 + height * width, bsz):
                idxe = min(idx0 + bsz, idx0_camera + height * width)
                idxs = list(range(idx0, idxe))
                origins, directions, _ = ray_dataset[idxs]
                with torch_inference_mode():
                    rgb_model = self._nerf_model(origins, directions) * 255.0
                    img[idx0 - idx0_camera : idxe - idx0_camera] = rgb_model
            idx0 = idxe
            img = img.reshape(height, width, -1)  # (H, W, C)
            imgs.append(img)
        return imgs
