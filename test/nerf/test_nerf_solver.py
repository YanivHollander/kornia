from pathlib import Path
from test.nerf.test_data_utils import create_random_images_for_cameras, create_red_images_for_cameras
from test.nerf.test_rays import create_four_cameras, create_one_camera

import pytest
import torch

from kornia.nerf.nerf_solver import NerfParams, NerfSolver
from kornia.testing import assert_close


@pytest.fixture
def checkpoint_path():
    return Path(__file__).parent / './checkpoint.tar'


class TestNerfSolver:
    def test_initialization(self, device, dtype):
        # Normal initialization
        nerf_obj = NerfSolver(device, dtype)
        cameras = create_four_cameras(device, dtype)
        imgs = create_random_images_for_cameras(cameras)
        nerf_obj.set_cameras_and_images_for_training(cameras=cameras, imgs=imgs)
        try:
            nerf_obj.run(num_iters=1)
        except Exception:
            pytest.fail('NeRF object failed to rum')

        # No cameras
        nerf_obj.set_cameras_and_images_for_training(cameras=None, imgs=imgs)
        with pytest.raises(TypeError, match='Invalid camera object'):
            nerf_obj.run(num_iters=1)

        # No images
        nerf_obj.set_cameras_and_images_for_training(cameras=cameras, imgs=None)
        with pytest.raises(TypeError, match='Invalid image list object'):
            nerf_obj.run(num_iters=1)

        # Fewer images than cameras
        nerf_obj.set_cameras_and_images_for_training(cameras=cameras, imgs=imgs[:-1])
        with pytest.raises(ValueError, match='Number of cameras must match number of input images'):
            nerf_obj.run(num_iters=1)

        # Number of image channels is not 3 for all images
        imgs[0] = imgs[0][:2, ...]
        nerf_obj.set_cameras_and_images_for_training(cameras=cameras, imgs=imgs)
        with pytest.raises(ValueError, match='All images must have three RGB channels'):
            nerf_obj.run(num_iters=1)

        # Height discrepancy
        imgs = create_random_images_for_cameras(cameras)
        imgs[0] = imgs[0][:, :-1, ...]
        nerf_obj.set_cameras_and_images_for_training(cameras=cameras, imgs=imgs)
        with pytest.raises(ValueError, match='All image heights must match camera heights'):
            nerf_obj.run(num_iters=1)

        # Width discrepancy
        imgs = create_random_images_for_cameras(cameras)
        imgs[0] = imgs[0][..., :-1]
        nerf_obj.set_cameras_and_images_for_training(cameras=cameras, imgs=imgs)
        with pytest.raises(ValueError, match='All image widths must match camera widths'):
            nerf_obj.run(num_iters=1)

    def test_parameter_change_after_one_epoch(self, device, dtype):
        torch.manual_seed(1)  # For reproducibility of random processes
        nerf_obj = NerfSolver(device, dtype)
        cameras = create_four_cameras(device, dtype)
        imgs = create_random_images_for_cameras(cameras)
        nerf_obj.set_cameras_and_images_for_training(cameras=cameras, imgs=imgs)

        params_before_update = [torch.clone(param).detach() for param in nerf_obj.nerf_model.parameters()]

        nerf_obj.run(num_iters=1)

        params_after_update = [torch.clone(param).detach() for param in nerf_obj.nerf_model.parameters()]

        assert all(
            not torch.equal(param_before_update, param_after_update)
            for param_before_update, param_after_update in zip(params_before_update, params_after_update)
        )

    def test_only_red(self, device, dtype):
        torch.manual_seed(1)  # For reproducibility of random processes
        camera = create_one_camera(5, 9, device, dtype)
        img = create_red_images_for_cameras(camera, device)

        nerf_obj = NerfSolver(device, dtype)
        nerf_obj.set_cameras_and_images_for_training(cameras=camera, imgs=img)
        nerf_obj.run(num_iters=35)

        torch.manual_seed(2)  # Reset seed for rendering result reproducibility
        img_rendered = nerf_obj.render_views(camera)[0].permute(2, 0, 1)

        assert_close(img_rendered.to(device, dtype) / 255.0, img[0].to(device, dtype) / 255.0, rtol=1.0e-5, atol=0.01)

    def test_single_ray(self, device, dtype):
        camera = create_one_camera(5, 9, device, dtype)
        img = create_red_images_for_cameras(camera, device)

        nerf_params = NerfParams(batch_size=1)
        nerf_obj = NerfSolver(device, dtype, params=nerf_params)
        nerf_obj.set_cameras_and_images_for_training(cameras=camera, imgs=img)
        nerf_obj.run(num_iters=20)

    def test_save_and_load_checkpoint(self, device, dtype, checkpoint_path):
        nerf_obj = NerfSolver(device, dtype)
        nerf_obj.save_checkpoint(checkpoint_path)

        nerf_obj_new = NerfSolver(device, dtype)
        nerf_obj_new.load_checkpoint(checkpoint_path)

        assert nerf_obj._iter == nerf_obj_new._iter
        assert nerf_obj._params == nerf_obj_new._params

        weights = [torch.clone(weight).detach() for weight in nerf_obj.nerf_model.parameters()]
        weights_new = [torch.clone(weight).detach() for weight in nerf_obj_new.nerf_model.parameters()]
        assert all(torch.equal(weight, weight_new) for weight, weight_new in zip(weights, weights_new))

        assert nerf_obj._opt_nerf.state_dict() == nerf_obj_new._opt_nerf.state_dict()
