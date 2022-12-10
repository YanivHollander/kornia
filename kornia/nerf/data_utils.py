from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler
from typing_extensions import TypeGuard

from kornia.core import Device, Tensor, stack
from kornia.geometry.camera import PinholeCamera
from kornia.io import ImageLoadType, load_image
from kornia.nerf.camera_utils import cameras_for_ids
from kornia.nerf.core import Images, ImageTensors
from kornia.nerf.rays import RandomRaySampler, RaySampler, UniformRaySampler


def _check_image_type_consistency(imgs: Images) -> None:
    if not all(isinstance(img, str) for img in imgs) and not all(isinstance(img, Tensor) for img in imgs):
        raise ValueError('The list of input images can only be all paths or tensors')


def _check_camera_image_consistency(cameras: PinholeCamera, imgs: ImageTensors) -> None:
    if len(imgs) != cameras.batch_size:
        raise ValueError(f'Number of images {len(imgs)} does not match number of cameras {cameras.batch_size}')
    if not all(img.shape[0] == 3 for img in imgs):
        raise ValueError('Not all input images have 3 channels')
    for i, (img, height, width) in enumerate(zip(imgs, cameras.height, cameras.width)):
        if img.shape[1:] != (height, width):
            raise ValueError(
                f'Image index {i} dimensions {(img.shape[1], img.shape[2])} are inconsistent with equivalent '
                f'camera dimensions {(height.item(), width.item())}'
            )


RayGroup = Tuple[Tensor, Tensor, Optional[Tensor]]


def _is_list_of_str(lst: Sequence[object]) -> TypeGuard[List[str]]:
    return isinstance(lst, list) and all(isinstance(x, str) for x in lst)


def _is_list_of_tensors(lst: Sequence[object]) -> TypeGuard[List[Tensor]]:
    return isinstance(lst, list) and all(isinstance(x, Tensor) for x in lst)


class RayDataset(Dataset):
    r"""Class to represent a dataset of rays.

    Args:
        cameras: scene cameras: PinholeCamera
        min_depth: sampled rays minimal depth from cameras: float
        max_depth: sampled rays maximal depth from cameras: float
        ndc: convert ray parameters to normalized device coordinates: bool
        device: device for ray tensors: Union[str, torch.device]
        dtype: type of ray tensors: torch.dtype
    """

    def __init__(
        self, cameras: PinholeCamera, min_depth: float, max_depth: float, ndc: bool, device: Device, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self._ray_sampler: Optional[RaySampler] = None
        self._imgs: Optional[List[Tensor]] = None
        self._cameras = cameras
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._ndc = ndc
        self._device = device
        self._dtype = dtype

    def init_ray_dataset(self, num_img_rays: Optional[Tensor] = None) -> None:
        r"""Initializes a ray dataset.

        Args:
            num_img_rays: If not None, number of rays to randomly cast from each camera: math: `(B)`.
        """
        if num_img_rays is None:
            self._init_uniform_ray_dataset()
        else:
            self._init_random_ray_dataset(num_img_rays)

    def init_images_for_training(self, imgs: Images) -> None:
        r"""Initializes images for training. Images can be either a list of tensors, or a list of paths to image
        disk locations.

        Args:
            imgs: List of image tensors or image paths: Images
        """
        _check_image_type_consistency(imgs)

        if _is_list_of_str(imgs):  # Load images from disk
            images = self._load_images(imgs)
        elif _is_list_of_tensors(imgs):
            images = imgs  # Take images provided on input
        else:
            raise TypeError(f'Expected a list of image tensors or image paths. Gotcha {type(imgs)}.')

        _check_camera_image_consistency(self._cameras, images)

        # Move images to defined device
        self._imgs = [img.to(self._device) for img in images]

    def _init_random_ray_dataset(self, num_img_rays: Tensor) -> None:
        r"""Initializes a random ray sampler and calculates dataset ray parameters.

        Args:
            num_img_rays: If not None, number of rays to randomly cast from each camers: math: `(B)`.
        """
        self._ray_sampler = RandomRaySampler(
            self._min_depth, self._max_depth, self._ndc, device=self._device, dtype=self._dtype
        )
        self._ray_sampler.calc_ray_params(self._cameras, num_img_rays)

    def _init_uniform_ray_dataset(self) -> None:
        r"""Initializes a uniform ray sampler and calculates dataset ray parameters."""
        self._ray_sampler = UniformRaySampler(
            self._min_depth, self._max_depth, self._ndc, device=self._device, dtype=self._dtype
        )
        self._ray_sampler.calc_ray_params(self._cameras)

    @staticmethod
    def _load_images(img_paths: List[str]) -> List[Tensor]:
        imgs: List[Tensor] = []
        for img_path in img_paths:
            imgs.append(load_image(img_path, ImageLoadType.UNCHANGED))
        return imgs

    def __len__(self):
        return len(self._ray_sampler)

    def __getitem__(self, idxs: Union[int, List[int]]) -> RayGroup:
        r"""Gets a dataset item.

        Args:
            idxs: An index or group of indices of ray parameter object: Union[int, List[int]]

        Return:
            A ray parameter object that includes ray origins, directions, and rgb values at the ray 2d pixel
            coordinates: RayGroup
        """
        if not isinstance(self._ray_sampler, RaySampler):
            raise TypeError('Ray sampler is not initiate yet, please run self.init_ray_dataset() before use it.')

        origins = self._ray_sampler.origins[idxs]
        directions = self._ray_sampler.directions[idxs]
        if self._imgs is None:
            return origins, directions, None
        camera_ids = self._ray_sampler.camera_ids[idxs]
        points_2d = self._ray_sampler.points_2d[idxs]
        rgbs = None
        imgs_for_ids = [self._imgs[i] for i in camera_ids]
        rgbs = stack([img[:, point2d[1].item(), point2d[0].item()] for img, point2d in zip(imgs_for_ids, points_2d)])
        rgbs = rgbs.to(dtype=self._dtype) / 255.0
        return origins, directions, rgbs


def instantiate_ray_dataloader(dataset: RayDataset, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
    r"""Initializes a dataloader to manage a ray dataset.

    Args:
        dataset: A ray dataset: RayDataset
        batch_size: Number of rays to sample in a batch: int
        shuffle: Whether to shuffle rays or sample then sequentially: bool
    """

    def collate_rays(items: List[RayGroup]) -> RayGroup:
        return items[0]

    if TYPE_CHECKING:
        # TODO: remove the type ignore when kornia relies on kornia 1.10
        return DataLoader(dataset)
    else:
        return DataLoader(
            dataset,
            sampler=BatchSampler(
                RandomSampler(dataset) if shuffle else SequentialSampler(dataset), batch_size, drop_last=False
            ),
            collate_fn=collate_rays,
        )


class RandomBatchRayDataset:
    def __init__(
        self,
        cameras: PinholeCamera,
        batch_size: int,
        min_depth: float,
        max_depth: float,
        device: Device,
        dtype: torch.dtype,
        imgs: Optional[ImageTensors] = None,
    ) -> None:
        self._cameras = cameras
        self._batch_size = batch_size
        sizes = [height * width for (height, width) in zip(cameras.height, cameras.width)]
        self._cum_sizes = torch.cumsum(torch.tensor(sizes).to(dtype=torch.int32, device=device), dim=0)
        self._ray_sampler = RandomRaySampler(min_depth, max_depth, False, device, dtype)
        self._device = device
        self._dtype = dtype
        if imgs is not None:
            _check_image_type_consistency(imgs)
            _check_camera_image_consistency(cameras, imgs)
        self._imgs = imgs

    def __len__(self):
        return self._cum_sizes[-1].item()

    def get_batch(self) -> RayGroup:
        idxs = torch.randperm(self.__len__(), device=self._device)[: self._batch_size]
        non_unique_camera_ids = torch.searchsorted(self._cum_sizes, idxs, right=True)
        camera_ids, num_img_rays = torch.unique(non_unique_camera_ids, return_counts=True)
        cameras = cameras_for_ids(self._cameras, camera_ids)
        self._ray_sampler.calc_ray_params(cameras, num_img_rays)
        origins = self._ray_sampler.origins
        directions = self._ray_sampler.directions
        rgbs = None
        if self._imgs is not None:
            camera_ids = self._ray_sampler.camera_ids
            points_2d = self._ray_sampler.points_2d
            rgbs = torch.stack(
                [self._imgs[i][:, point2d[1].item(), point2d[0].item()] for i, point2d in zip(camera_ids, points_2d)]
            )
            rgbs = rgbs.to(dtype=self._dtype) / 255.0
        idxs = torch.randperm(min(self._batch_size, self.__len__()))
        return origins[idxs], directions[idxs], rgbs[idxs] if rgbs is not None else None
