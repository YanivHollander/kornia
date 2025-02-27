from __future__ import annotations

from kornia.core import Module, Tensor

from .filter import filter2d
from .kernels import get_box_kernel2d, normalize_kernel2d


def box_blur(
    input: Tensor, kernel_size: tuple[int, int] | int, border_type: str = 'reflect', normalized: bool = True
) -> Tensor:
    r"""Blur an image using the box filter.

    .. image:: _static/img/box_blur.png

    The function smooths an image using the kernel:

    .. math::
        K = \frac{1}{\text{kernel_size}_x * \text{kernel_size}_y}
        \begin{bmatrix}
            1 & 1 & 1 & \cdots & 1 & 1 \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
        \end{bmatrix}

    Args:
        image: the image to blur with shape :math:`(B,C,H,W)`.
        kernel_size: the blurring kernel size.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        normalized: if True, L1 norm of the kernel is set to 1.

    Returns:
        the blurred tensor with shape :math:`(B,C,H,W)`.

    .. note::
       See a working example `here <https://kornia-tutorials.readthedocs.io/en/latest/
       filtering_operators.html>`__.

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> output = box_blur(input, (3, 3))  # 2x4x5x7
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """
    kernel = get_box_kernel2d(kernel_size, device=input.device, dtype=input.dtype)
    if normalized:
        kernel = normalize_kernel2d(kernel)
    return filter2d(input, kernel, border_type)


class BoxBlur(Module):
    r"""Blur an image using the box filter.

    The function smooths an image using the kernel:

    .. math::
        K = \frac{1}{\text{kernel_size}_x * \text{kernel_size}_y}
        \begin{bmatrix}
            1 & 1 & 1 & \cdots & 1 & 1 \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
            \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
            1 & 1 & 1 & \cdots & 1 & 1 \\
        \end{bmatrix}

    Args:
        kernel_size: the blurring kernel size.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``. Default: ``'reflect'``.
        normalized: if True, L1 norm of the kernel is set to 1.

    Returns:
        the blurred input tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Example:
        >>> input = torch.rand(2, 4, 5, 7)
        >>> blur = BoxBlur((3, 3))
        >>> output = blur(input)  # 2x4x5x7
        >>> output.shape
        torch.Size([2, 4, 5, 7])
    """

    def __init__(
        self, kernel_size: tuple[int, int] | int, border_type: str = 'reflect', normalized: bool = True
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.border_type = border_type
        self.normalized = normalized

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"normalized={self.normalized}, "
            f"border_type={self.border_type})"
        )

    def forward(self, input: Tensor) -> Tensor:
        return box_blur(input, self.kernel_size, self.border_type, self.normalized)
