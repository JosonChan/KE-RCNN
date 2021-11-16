
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn.utils import kaiming_init
from mmdet.models.builder import HEADS

from .bbox_head import BBoxHead
from ...utils.token_transformer_block import FeatEmbed


@HEADS.register_module()
class RelationHeadv4(BBoxHead):
    def __init__(self,
                 num_classes=294,
                 num_bbox_classes=47,
                 in_dims=256,
                 bbox_cls_fc_dims=1024,
                 patch_size=2,
                 human_roi_feat_area=28*28,
                 cross_encoder=None,
                 self_decoder=None,
                 cross_decoder=None,
                 *arg, **kwargs):
        super(RelationHeadv4, self).__init__(*arg, **kwargs)

        self.in_dims = in_dims
        self.half_dims = in_dims // 2
        self.num_classes = num_classes
        self.num_bbox_classes = num_bbox_classes

        # bbox embedding
        self.bbox_conv = nn.Sequential(
            nn.Conv2d(self.half_dims, self.half_dims, kernel_size=3, padding=1),
            nn.GroupNorm(4, self.half_dims),
            nn.ReLU())
        # gt_bbox embedding
        self.gt_bbox_embed = nn.Sequential(
            nn.Conv2d(self.in_dims, self.half_dims, kernel_size=1),
            nn.GroupNorm(4, self.half_dims),
            nn.ReLU())

        # encoder
        self.encoder = build_transformer_layer_sequence(cross_encoder)
        encoder_pos_embed = self._make_sine_position_embedding(self.roi_feat_area, self.half_dims)
        human_encoder_pos_embed = self._make_sine_position_embedding(human_roi_feat_area, self.half_dims)
        self.encoder_pos_embed = nn.Parameter(encoder_pos_embed, requires_grad=False)
        self.human_encoder_pos_embed = nn.Parameter(human_encoder_pos_embed, requires_grad=False)

        # knowleage matrix
        knowledge_matrix = np.load('data/fashionpedia/train_attr_knowledge_matrix.npy')
        knowledge_matrix_tmp = knowledge_matrix.copy()
        inds = (knowledge_matrix == 0)
        knowledge_matrix_tmp[inds] = np.inf

        knowledge_matrix = (knowledge_matrix - knowledge_matrix_tmp.min(axis=1)[:, None]) / \
            (knowledge_matrix.max(axis=1)[:, None] - knowledge_matrix_tmp.min(axis=1)[:, None] + 1e-6)
        knowledge_matrix[inds] = 0
        self.knowledge_matrix = nn.Parameter(
            torch.tensor(knowledge_matrix, dtype=torch.float32), requires_grad=False)
        
        # fc_attr_cls_weight
        self.fc_attr_cls_weight = nn.Parameter(torch.randn(self.num_classes, in_dims))
        kaiming_init(self.fc_attr_cls_weight)

        # for no used paramter
        del self.fc_cls
        self.init_cfg = []

        # ffn
        self.fc_attr_transform = nn.Sequential(
            nn.Linear(in_dims, in_dims),
            nn.LayerNorm((in_dims)),
            nn.ReLU(),
            nn.Linear(in_dims, in_dims),
            nn.LayerNorm((in_dims)))

        # bbox embed
        self.bbox_embed = FeatEmbed(self.roi_feat_size, patch_size)

        # bbox loc embed
        self.loc_embed = nn.Linear(4, in_dims)

        # bbox cls embed
        self.cls_embed = nn.Sequential(
            nn.Linear(bbox_cls_fc_dims, in_dims),
            nn.ReLU(),
            nn.Linear(in_dims, in_dims))

        # pos embedding
        bbox_pos_embed = self._make_sine_position_embedding(self.bbox_embed.num_patches, in_dims)
        other_pos_embed = torch.zeros((1, 1 + self.num_classes + self.num_classes, in_dims))
        self.pos_embed = nn.Parameter(torch.cat([bbox_pos_embed, other_pos_embed], dim=1), requires_grad=False)

        # decoder
        self.self_decoder = build_transformer_layer_sequence(self_decoder)
        self.cross_decoder = build_transformer_layer_sequence(cross_decoder)
    
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
                human_bbox_feats,
                pos_gt_labels,
                pos_cls_scores,
                bbox_locs,
                fc_cls_weight):
        
        # bbox feats embed
        b, c, h, w = bbox_feats.shape
        bbox_src_feats = bbox_feats[:, :c//2, :, :]
        bbox_src_feats = self.bbox_conv(bbox_src_feats)
        human_bbox_feats = self.gt_bbox_embed(human_bbox_feats)

        # qkv
        query = bbox_feats[:, c//2:, :, :].flatten(2).permute(2, 0, 1)
        key = value = human_bbox_feats.flatten(2).permute(2, 0, 1)

        # for bbox feat
        encoder_pos_embed = self.encoder_pos_embed.repeat(b, 1, 1).permute(1, 0, 2)
        human_encoder_pos_embed = self.human_encoder_pos_embed.repeat(b, 1, 1).permute(1, 0, 2)
        bbox_rela_feats = self.encoder(query=query,
                                       key=key,
                                       value=value,
                                       key_pos=human_encoder_pos_embed,
                                       query_pos=encoder_pos_embed)
        bbox_rela_feats = bbox_rela_feats.permute(1, 2, 0).reshape(b, self.half_dims, h, w)
        bbox_feats = torch.cat([bbox_src_feats, bbox_rela_feats], dim=1)
        bbox_feats = self.bbox_embed(bbox_feats).permute(1, 0, 2)

        # for bbox cls
        with torch.no_grad():
            bbox_cls_feats = pos_cls_scores.unsqueeze(-1) * fc_cls_weight.unsqueeze(0)
        knowledge_matrix = self.knowledge_matrix.unsqueeze(0).repeat(b, 1, 1)
        bbox_cls_feats = torch.matmul(
            knowledge_matrix.permute(0, 2, 1), bbox_cls_feats[:, :self.num_bbox_classes-1, :])
        bbox_cls_feats = self.cls_embed(bbox_cls_feats).permute(1, 0, 2)

        # decoder query
        attr_cls_weight = self.fc_attr_transform(self.fc_attr_cls_weight)
        human_knowledge = attr_cls_weight.new_zeros((1, 294), dtype=torch.float32)
        knowledge_matrix_all = torch.cat([self.knowledge_matrix, human_knowledge], dim=0)
        attr_knowledge = knowledge_matrix_all[pos_gt_labels]
        attr_knowledge = (attr_knowledge > 0).float()
        decoder_query = attr_knowledge.unsqueeze(-1) * attr_cls_weight.unsqueeze(0)
        decoder_query = decoder_query.permute(1, 0, 2)

        # for bbox location
        bbox_locs = self.loc_embed(bbox_locs).unsqueeze(0)

        # for input x
        x = torch.cat([bbox_feats, bbox_cls_feats, bbox_locs, decoder_query], dim=0)

        # pos embed
        pos_embed = self.pos_embed.repeat(b, 1, 1).permute(1, 0, 2)

        # transformer
        memory = self.self_decoder(
            query=x,
            key=None,
            value=None,
            key_pos=pos_embed,
            query_pos=pos_embed)
        cross_query_embed = memory[-self.num_classes:, :, :]

        out_dec = self.cross_decoder(
            query=cross_query_embed,
            key=memory[:-self.num_classes, :, :],
            value=memory[:-self.num_classes, :, :],
            key_pos=pos_embed[:-self.num_classes, :, :])
        out_dec = out_dec.permute(1, 0, 2)
        cls_score = self.fc_attr_cls_weight.unsqueeze(0) * out_dec

        return cls_score.sum(dim=-1)
