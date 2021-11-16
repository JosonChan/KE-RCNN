from timm.models.layers.weight_init import trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from einops import rearrange

from mmcv.cnn import build_conv_layer, kaiming_init


class FeatEmbed(nn.Module):
    """Image to Patch Embedding.

    Args:
        img_size (int | tuple): Size of input image.
        patch_size (int): Size of one patch.
        in_channels (int): Channel num of input features. Defaults to 3.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        conv_cfg (dict | None): Config dict for convolution layer. Defaults to
            `dict(type='Conv2d')`.
    """

    def __init__(self,
                 img_size,
                 patch_size,
                 in_channels=256,
                 embed_dims=256,
                 conv_cfg=dict(type='Conv2d')):
        super().__init__()
        self.img_size = _pair(img_size)
        self.patch_size = _pair(patch_size)

        num_patches = (self.img_size[1] // self.patch_size[1]) * (
            self.img_size[0] // self.patch_size[0])
        assert num_patches * self.patch_size[0] * self.patch_size[1] == \
               self.img_size[0] * self.img_size[1], \
               'The image size H*W must be divisible by patch size'
        self.num_patches = num_patches

        # Use conv layer to embed
        self.projection = build_conv_layer(
            conv_cfg,
            in_channels,
            embed_dims,
            kernel_size=patch_size,
            stride=patch_size)

        self.init_weights()

    def init_weights(self):
        # Lecun norm from ClassyVision
        kaiming_init(self.projection, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        x = self.projection(x).flatten(2)
        x = rearrange(x, 'b d n -> b n d')
        return x
