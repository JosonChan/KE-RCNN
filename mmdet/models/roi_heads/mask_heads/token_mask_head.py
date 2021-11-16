import torch
import torch.nn as nn
import math

from mmcv.cnn import ConvModule, Linear, xavier_init
from mmcv.runner import ModuleList, auto_fp16, base_module
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.core import mask_target

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils import build_transformer
from mmdet.models.utils import FeatEmbed


@HEADS.register_module()
class TokenMaskHead(BaseModule):
    def __init__(self,
                 num_classes,
                 bbox_in_channels,
                 bbox_feat_size,
                 human_in_channels,
                 human_feat_size,
                 mask_size=28,
                 encoder=None,
                 decoder=None,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwarg):
        super(TokenMaskHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.bbox_in_channels = bbox_in_channels
        self.bbox_feat_size = bbox_feat_size
        self.human_in_channels = human_in_channels
        self.human_feat_size = human_feat_size
        self.mask_size = mask_size
        self.mask_size_dim = int(mask_size * mask_size)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)

        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.encoder.embed_dims

        self._init_layers()

        bbox_pos_embed = self._make_sine_position_embedding(
                self.bbox_embed.num_patches, self.embed_dims)
        human_pos_embed = self._make_sine_position_embedding(
                self.human_embed.num_patches, self.embed_dims)
        other_pos_embed = torch.zeros((1, 2, 256), dtype=torch.float32)
        self.pos_embed = nn.Parameter(
            torch.cat([human_pos_embed, bbox_pos_embed, other_pos_embed], dim=1), requires_grad=False)
            
        self.num_patches = self.bbox_embed.num_patches + self.human_embed.num_patches
        self.num_patches += 2
    
    def _init_layers(self):
        self.query_embed = nn.Parameter(torch.randn(self.num_classes, 1, self.embed_dims)) 
        self.bbox_embed = FeatEmbed(img_size=self.bbox_feat_size,
                                    patch_size=3,
                                    in_channels=self.bbox_in_channels,
                                    embed_dims=self.embed_dims)
        self.human_embed = FeatEmbed(img_size=self.human_feat_size,
                                     patch_size=1,
                                     in_channels=self.human_in_channels,
                                     embed_dims=self.embed_dims)
        self.bbox_cls_embed = nn.Linear(self.num_classes, self.embed_dims)
        self.bbox_loc_embed = nn.Linear(4, self.embed_dims)
        self.mlp_head = nn.Sequential(
            nn.Linear(self.embed_dims, self.mask_size_dim),
            nn.LayerNorm(self.mask_size_dim),
            nn.GELU(),
            nn.Linear(self.mask_size_dim, self.mask_size_dim)
        )
    
    def init_weights(self):
         # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True
    
    def _make_sine_position_embedding(self, num_patches, dim, temperature=10000, scale=2*math.pi):
        
        h = w = int(num_patches**0.5)
        assert h * w == num_patches
        
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = dim // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def forward(self,
                bbox_feats,
                human_feats,
                bbox_cls,
                bbox_loc,
                img_metas,
                cfg=None):
        
        # get info
        bs, c, h, w = bbox_feats.shape
        # human embed
        human_token = self.human_embed(human_feats)
        # bbox embed
        bbox_token = self.bbox_embed(bbox_feats)
        # bbox cls embed
        cls_token = self.bbox_cls_embed(bbox_cls).unsqueeze(1)
        # bbox loc embed
        loc_token = self.bbox_loc_embed(bbox_loc).unsqueeze(1)
        # feats concate
        x = torch.cat([human_token, bbox_token, cls_token, loc_token], dim=1).permute(1, 0, 2)
        # pos embed
        pos_embed = self.pos_embed.repeat(bs, 1, 1).permute(1, 0, 2)
        # masks
        mask = x.new_zeros(bs, self.num_patches, dtype=torch.bool)
        # query embed
        query_embed = self.query_embed.repeat(1, bs, 1)
        # transformer
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask)
        target = torch.zeros_like(query_embed)
        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask)
        mask_pred = self.mlp_head(out_dec)
        return mask_pred.squeeze(0).permute(1, 0, 2).reshape(bs, -1, self.mask_size, self.mask_size)
    
    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        max_pos_per_img = rcnn_train_cfg.max_pos_per_img
        pos_proposals = [res.pos_bboxes[:max_pos_per_img, :] 
                         if res.pos_bboxes.shape[0] > max_pos_per_img
                         else res.pos_bboxes 
                         for res in sampling_results]

        pos_assigned_gt_inds = [res.pos_assigned_gt_inds[:max_pos_per_img]
                                if res.pos_assigned_gt_inds.shape[0] > max_pos_per_img
                                else res.pos_assigned_gt_inds
                                for res in sampling_results]

        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets
    
    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        """
        Example:
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> # There are lots of variations depending on the configuration
            >>> self = FCNMaskHead(num_classes=C, num_convs=1)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> sf = self.scale_factor
            >>> labels = torch.randint(0, C, size=(N,))
            >>> # With the default properties the mask targets should indicate
            >>> # a (potentially soft) single-class label
            >>> mask_targets = torch.rand(N, H * sf, W * sf)
            >>> loss = self.loss(mask_pred, mask_targets, labels)
            >>> print('loss = {!r}'.format(loss))
        """
        loss = dict()
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_pred, mask_targets,
                                           torch.zeros_like(labels))
            else:
                loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss  
