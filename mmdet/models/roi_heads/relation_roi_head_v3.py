# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np

from mmdet.core import bbox, bbox2roi
from mmdet.core import (bbox2roi, bbox_mapping)

from .standard_roi_head import StandardRoIHead
from ..builder import HEADS, build_roi_extractor


@HEADS.register_module()
class RelationRoIHeadv3(StandardRoIHead):

    def __init__(self, max_pos_per_img=32,
                 human_bbox_roi_extractor=None, **kwargs):
        super(RelationRoIHeadv3, self).__init__(**kwargs)
        self.human_bbox_roi_extractor = build_roi_extractor(human_bbox_roi_extractor)
        self.max_pos_per_img = max_pos_per_img

    def forward_train(self,
                      x,
                      img_metas,
                      sampling_results,
                      bbox_results,
                      gt_bboxes,
                      gt_labels,
                      fc_cls_weight=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        gt_attributes = kwargs['gt_attributes']

        losses = dict()
        attr_results = self._attr_forward_train(x, 
                                                sampling_results,
                                                bbox_results,
                                                gt_bboxes,
                                                gt_labels,
                                                gt_attributes,
                                                fc_cls_weight,
                                                img_metas)
        losses.update(attr_results['loss_attr'])

        return losses

    def _bbox_forward(self,
                      bbox_feats_list,
                      gt_human_bbox_feats_list,
                      pos_cls_score_list,
                      bbox_locs_list,
                      fc_cls_weight):

        cls_score = self.bbox_head(bbox_feats_list,
                                   gt_human_bbox_feats_list,
                                   pos_cls_score_list,
                                   bbox_locs_list,
                                   fc_cls_weight)

        bbox_results = dict(
            cls_score=cls_score)
        return bbox_results

    def _attr_forward_train(self,
                            x,
                            sampling_results,
                            bbox_results,
                            gt_bboxes,
                            gt_labels,
                            gt_attributes,
                            fc_cls_weight,
                            img_metas):

        # get pos proposals and filter human proposals
        pos_proposals_list = []
        bboxes_num_list = []
        pos_bboxes_num_list = []
        inds_list = []
        for res in sampling_results:
            pos_bboxes = res.pos_bboxes
            inds = (res.pos_gt_labels != self.train_cfg.num_classes-1)
            # max_num_bbox filter
            if pos_bboxes.shape[0] > self.max_pos_per_img:
                pos_bboxes = pos_bboxes[:self.max_pos_per_img, :]
                inds = inds[:self.max_pos_per_img]

            pos_proposals_list.append(pos_bboxes[inds])
            bboxes_num_list.append(res.bboxes.shape[0])
            pos_bboxes_num_list.append(pos_bboxes.shape[0])
            inds_list.append(inds)
        pos_proposals = torch.cat(pos_proposals_list, dim=0)

        # get the pos_cls_cores
        with torch.no_grad():
            pos_cls_score_list = []
            cls_scores = bbox_results['cls_score'].split(bboxes_num_list, dim=0)
            for i, pos_num in enumerate(pos_bboxes_num_list):
                pos_cls_score = cls_scores[i][:pos_num]
                pos_cls_score = pos_cls_score[inds_list[i]]
                pos_cls_score = pos_cls_score.softmax(dim=-1)
                pos_cls_score_list.append(pos_cls_score)
            pos_cls_score = torch.cat(pos_cls_score_list, dim=0)
 
        # get the human feats
        gt_human_bbox_list = []
        gt_human_bbox_feats_list = []
        for i in range(len(sampling_results)):
            gt_bbox = gt_bboxes[i]
            human_bbox = gt_bbox[-1, :].reshape(-1, 4)
            gt_human_bbox_list.append(human_bbox)
            human_bbox_roi = bbox2roi([human_bbox])
            human_bbox_feat = self.human_bbox_roi_extractor(
                x[:self.human_bbox_roi_extractor.num_inputs], human_bbox_roi)
            b, d, h, w = human_bbox_feat.shape
            human_bbox_feat = human_bbox_feat.repeat(pos_proposals_list[i].shape[0], 1, 1, 1)
            gt_human_bbox_feats_list.append(human_bbox_feat)
        gt_human_bbox_feats = torch.cat(gt_human_bbox_feats_list, dim=0)

        # get the bbox relation
        with torch.no_grad():
            bbox_locs_list = []
            for i, res in enumerate(gt_human_bbox_list):
                gt_human_bbox = res.repeat(pos_proposals_list[i].shape[0], 1)
                bbox_locs = self.bbox_head.bbox_coder.encode(pos_proposals_list[i], gt_human_bbox)
                bbox_locs_list.append(bbox_locs)
            bbox_locs = torch.cat(bbox_locs_list, dim=0)

        rois = bbox2roi(pos_proposals_list)
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        
        # forward
        attr_results = self._bbox_forward(bbox_feats,
                                          gt_human_bbox_feats,
                                          pos_cls_score,
                                          bbox_locs,
                                          fc_cls_weight)

        # get the proposals_gt_attributes and filter human proposals
        proposals_gt_attributes_list = []
        for i, res in enumerate(sampling_results):
            pos_assigned_gt_inds = res.pos_assigned_gt_inds
            if res.pos_assigned_gt_inds.shape[0] > self.max_pos_per_img:
                pos_assigned_gt_inds = pos_assigned_gt_inds[:self.max_pos_per_img]
            pos_assigned_gt_inds = pos_assigned_gt_inds[inds_list[i]]
            proposals_gt_attributes = gt_attributes[i][pos_assigned_gt_inds].reshape(-1, 1)
            proposals_gt_attributes_list.append(proposals_gt_attributes)
        proposals_gt_attributes = torch.cat(proposals_gt_attributes_list, dim=0)

        # get target and loss
        attr_targets = self.get_targets(pos_proposals, proposals_gt_attributes, self.train_cfg)
        loss_attr = self.bbox_head.loss(attr_results['cls_score'].reshape(-1, 1), None, rois,
                                        *attr_targets)
        loss_attr_cls = dict(
            loss_attr=loss_attr['loss_cls'],
            attr_acc=loss_attr['acc'])

        attr_results.update(loss_attr=loss_attr_cls)

        return attr_results

    def get_targets(self, 
                    pos_proposals, 
                    proposals_gt_attributes, 
                    train_cfg=None):
        labels = proposals_gt_attributes.reshape(-1)
        label_weights = labels.new_ones(labels.size(0))

        return labels, label_weights, None, None
    
    def aug_test(self,
                 x,
                 garments_bboxes,
                 garments_scores,
                 garments_labels,
                 fc_cls_weight,
                 img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        attributes = self.aug_test_bboxes(x, img_metas,
                                             garments_bboxes,
                                             garments_scores,
                                             garments_labels,
                                             fc_cls_weight,
                                             self.test_cfg)
        garments_labels = garments_labels.detach().cpu().numpy()

        attribute_results = []
        for i in range(self.test_cfg.num_classes):
            ids = np.argwhere(garments_labels == i).reshape(-1)
            attribute_results.append([attributes[id] for id in ids])
        return attribute_results

    def aug_test_bboxes(self,
                        feats,
                        img_metas,
                        garments_bboxes,
                        garments_scores,
                        garments_labels,
                        fc_cls_weight,
                        rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            # TODO more flexible
            garments_proposals = bbox_mapping(garments_bboxes[:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            rois = bbox2roi([garments_proposals])
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)

            # get pos cls score
            pos_cls_score = garments_scores

            # get the human feats
            inds = torch.where(garments_labels == rcnn_test_cfg.num_classes-1)[0]
            max_ind = inds[torch.argmax(garments_bboxes[inds][:, 4])].reshape(-1,)
            human_rois = bbox2roi([garments_proposals[max_ind]])
            human_feats = self.human_bbox_roi_extractor(
                x[:self.human_bbox_roi_extractor.num_inputs], human_rois)
            human_feats = human_feats.repeat(garments_proposals.shape[0], 1, 1, 1)

            # get the human loc relation
            human_bbox = garments_proposals[max_ind]
            human_bbox = human_bbox.repeat(garments_proposals.shape[0], 1).reshape(-1, 4)

            bbox_locs = self.bbox_head.bbox_coder.encode(garments_proposals, human_bbox)

            # forward
            bbox_results = self._bbox_forward(bbox_feats,
                                              human_feats,
                                              pos_cls_score,
                                              bbox_locs,
                                              fc_cls_weight)
            scores = self.get_bboxes(bbox_results['cls_score'], cfg=rcnn_test_cfg)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_scores = self.merge_aug_bboxes(aug_scores, img_metas, rcnn_test_cfg)

        attributes = []
        for merged_score in merged_scores:
            attributes.append(np.argwhere(merged_score >= rcnn_test_cfg.attribute_score_thr).reshape(-1,))
        
        return attributes
    
    def get_bboxes(self, 
                   cls_score,  
                   cfg=None):
        scores = torch.sigmoid(cls_score).reshape(-1, cfg.attribute_num)

        return scores
    
    def merge_aug_bboxes(self, aug_scores, img_metas, rcnn_test_cfg):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).
            rcnn_test_cfg (dict): rcnn test config.

        Returns:
            tuple: (bboxes, scores)
        """
        scores = torch.stack(aug_scores).mean(dim=0).reshape(-1, rcnn_test_cfg.attribute_num)
        scores = scores.detach().cpu().numpy()

        return scores
