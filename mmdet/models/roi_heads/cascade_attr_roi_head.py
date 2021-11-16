import torch
import numpy as np

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from mmdet.core.bbox import bbox_mapping_back
from .cascade_roi_head import CascadeRoIHead
from ..builder import HEADS, build_head, build_roi_extractor


@HEADS.register_module()
class CascadeAttrRoIHead(CascadeRoIHead):
    def __init__(self, *arg, **kwargs):
        super(CascadeAttrRoIHead, self).__init__(*arg, **kwargs)
    
    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)
    
    def _attr_bbox_forward(self, stage, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(
            x[:bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, attr_score = bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, 
            bbox_pred=bbox_pred,
            attr_score=attr_score,
            bbox_feats=bbox_feats)
        return bbox_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results
    
    def aug_test_attr(self, x, proposal_list, img_metas, rescale=False, **kwargs):

        det_bboxes, det_labels, det_attr_scores, det_scores = \
            self.aug_test_bboxes(x, img_metas, proposal_list, self.test_cfg)

        det_attributes = []
        det_attr_scores = det_attr_scores.detach().cpu().numpy()
        for det_attr_score in det_attr_scores:
            det_attributes.append(np.argwhere(det_attr_score > self.test_cfg.attribute_score_thr).reshape(-1,))

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])

        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.test_cfg.num_classes)

        if self.with_mask == True:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
        else:
            segm_results = None

        attribute_results = []
        det_labels_tmp = det_labels.clone().detach().cpu().numpy()
        for i in range(self.test_cfg.num_classes):
            ids = np.argwhere(det_labels_tmp == i).reshape(-1)
            attribute_results.append([det_attributes[id] for id in ids])

        results = dict(
            det_results=[bbox_results],
            attr_results=attribute_results,
            segm_results=segm_results,
            garments_bboxes=_det_bboxes,
            garments_labels=det_labels,
            garments_scores=det_scores,
            det_attr_scores=det_attr_scores)
        return results
    
    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        aug_bboxes = []
        aug_scores = []
        aug_attr_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            rois = bbox2roi([proposals])
            
            ms_scores = []
            if rois.shape[0] == 0:
                # There is no proposal in the single image
                aug_bboxes.append(rois.new_zeros(0, 4))
                aug_scores.append(rois.new_zeros(0, 1))
                continue

            for i in range(self.num_stages):

                if i < self.num_stages - 1:
                    bbox_results = self._bbox_forward(i, x, rois)
                    ms_scores.append(bbox_results['cls_score'])
                    cls_score = bbox_results['cls_score']
                    if self.bbox_head[i].custom_activation:
                        cls_score = self.bbox_head[i].loss_cls.get_activation(
                            cls_score)
                    bbox_label = cls_score[:, :-1].argmax(dim=1)
                    rois = self.bbox_head[i].regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])
                else:
                    bbox_results = self._attr_bbox_forward(i, x, rois)
                    ms_scores.append(bbox_results['cls_score'])

            cls_score = sum(ms_scores) / float(len(ms_scores))

            bboxes, scores, attr_scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                bbox_results['attr_score'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
            aug_attr_scores.append(attr_scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores, merged_attr_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, aug_attr_scores, img_metas, rcnn_test_cfg)
        if merged_bboxes.shape[0] == 0:
            # There is no proposal in the single image
            det_bboxes = merged_bboxes.new_zeros(0, 5)
            det_labels = merged_bboxes.new_zeros((0, ), dtype=torch.long)
            det_attr_scores = merged_bboxes.new_zeros(0, rcnn_test_cfg.attribute_num)
        else:
            det_bboxes, det_labels, inds = multiclass_nms(merged_bboxes,
                                                          merged_scores,
                                                          rcnn_test_cfg.score_thr,
                                                          rcnn_test_cfg.nms,
                                                          rcnn_test_cfg.max_per_img,
                                                          return_inds=True)
            det_attr_scores = merged_attr_scores[:, None].expand(
                merged_attr_scores.size(0), rcnn_test_cfg.num_classes, rcnn_test_cfg.attribute_num)
            det_attr_scores = det_attr_scores.reshape(-1, rcnn_test_cfg.attribute_num)[inds]

            det_scores = merged_scores[inds // rcnn_test_cfg.num_classes]

        return det_bboxes, det_labels, det_attr_scores, det_scores
    
    def merge_aug_bboxes(self, aug_bboxes, aug_scores, aug_attr_scores, img_metas, rcnn_test_cfg):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).
            rcnn_test_cfg (dict): rcnn test config.

        Returns:
            tuple: (bboxes, scores)
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                    flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.stack(recovered_bboxes).mean(dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.stack(aug_scores).mean(dim=0)
        if aug_attr_scores is not None:
            aug_attr_scores = torch.stack(aug_attr_scores).mean(dim=0)
            return bboxes, scores, aug_attr_scores
