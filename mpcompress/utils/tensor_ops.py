import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
import PIL.Image as Image


def tensor2image(x: torch.Tensor) -> Image.Image:
    """
    Convert a tensor to a PIL Image.

    Clamps tensor values to [0, 1] range, removes dimensions of size 1,
    and converts to PIL Image format.

    Args:
        x (torch.Tensor): Input tensor with values in [0, 1] range.

    Returns:
        image (PIL.Image.Image): The converted PIL Image.
    """
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def center_pad(x, p):
    """
    Pad a tensor to make its spatial dimensions divisible by p, using center padding.

    Pads the tensor symmetrically (centered) so that both height and width
    become multiples of p. The padding is distributed evenly on both sides
    when possible, with any remainder added to the right/bottom.

    Args:
        x (torch.Tensor): Input tensor with shape (B, C, H, W).
        p (int): Padding divisor. Height and width will be padded to be divisible by p.

    Returns:
        x_padded (torch.Tensor): Padded tensor with shape (B, C, new_H, new_W).
        padding (tuple): Tuple of (padding_left, padding_right, padding_top, padding_bottom)
            that can be used with center_crop to restore original size.
    """
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def center_crop(x, padding):
    """
    Remove center padding from a tensor.

    Crops the tensor by removing the padding that was added by center_pad.
    Uses negative padding values to crop from the edges.

    Args:
        x (torch.Tensor): Padded tensor to crop.
        padding (tuple): Tuple of (padding_left, padding_right, padding_top, padding_bottom)
            returned by center_pad.

    Returns:
        x_cropped (torch.Tensor): Cropped tensor with original spatial dimensions restored.
    """
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )


def border_pad(x, patch_size):
    """
    Pad a tensor on the right and bottom borders to make dimensions divisible by patch_size.

    Adds padding only to the right (width) and bottom (height) edges,
    ensuring both dimensions become multiples of patch_size.

    Args:
        x (torch.Tensor): Input tensor with shape (B, C, H, W).
        patch_size (int): Patch size. Height and width will be padded to be divisible by patch_size.

    Returns:
        x_padded (torch.Tensor): Padded tensor with shape (B, C, new_H, new_W),
            where new_H and new_W are multiples of patch_size.
    """
    B, C, H, W = x.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
    return x
