kornia.nerf
===========

Neural Radiance Fields (NeRF) is a new and emerging method in computer vision for novel view synthesis of a 3D scene
based on a set of 2D images. The NeRF algorithm marches rays from random image pixels through the 3D scene. Sampled
points along the rays are mapped by a multi-layer perceptron (MLP) to color and volume density, which can then be rendered
to match the true color of the source pixel point. Training the NeRF model allows for a scene representation in a
continuous 3D space, which in turn enables the rendering of novel views.

Each ray point is represented by 5 parameters: 3 for position and two for direction.

.. image:: _static/img/nerf_overview.png


.. currentmodule:: kornia.nerf

.. toctree::
   :maxdepth: 3

   nerf.camera_utils
   nerf.core
   nerf.data_utils
   nerf.nerf_model
   nerf.nerf_solver
   nerf.positional_encoder
   nerf.rays
   nerf.renderer
