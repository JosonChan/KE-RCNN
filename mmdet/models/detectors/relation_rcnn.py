# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from ..builder import DETECTORS, build_head
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class RelationRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 attr_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(RelationRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        if train_cfg is not None:
            if isinstance(train_cfg.rcnn, list):
                rcnn_train_cfg = train_cfg.rcnn[-1]
            else:
                rcnn_train_cfg = train_cfg.rcnn
        else:
            rcnn_train_cfg = None
        attr_head.update(train_cfg=rcnn_train_cfg)
        attr_head.update(test_cfg=test_cfg.rcnn)
        attr_head.pretrained = pretrained
        self.attr_head = build_head(attr_head)
    
    def simple_test(self, img, img_meta, rescale, **kwargs):

        x = self.extract_feat(img)
        proposal_list = self.rpn_head.simple_test_rpn(x, img_meta)

        # garment forward
        results = self.roi_head.aug_test_attr([x],
                                              proposal_list,
                                              [img_meta],
                                              rescale=rescale,
                                              **kwargs)

        if isinstance(self.roi_head.bbox_head, nn.ModuleList):
            fc_cls_weight = self.roi_head.bbox_head[-1].get_fc_cls_weight()
        else:
            fc_cls_weight = self.roi_head.bbox_head.get_fc_cls_weight()

        # attribute forward
        attr_results_list = self.attr_head.aug_test([x], 
                                                    results['garments_bboxes'],
                                                    results['garments_scores'],
                                                    results['garments_labels'],
                                                    fc_cls_weight,
                                                    [img_meta], rescale=rescale)

        return [[results['det_results'], results['segm_results'], attr_results_list]]
