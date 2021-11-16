import mmcv
import numpy as np
from .builder import DATASETS
from .coco import CocoDataset
from mmdet.core import encode_mask_results

@DATASETS.register_module()
class FashionPedia(CocoDataset):

    CLASSES = ('shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan',
               'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress',
               'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory',
               'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings',
               'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar',
               'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 
               'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon',
               'rivet', 'ruffle', 'sequin', 'tassel')
    num_attribute = 294

    def __init__(self, with_human=False, *arg, **kwargs):
        super(FashionPedia, self).__init__(*arg, **kwargs)
        self.with_human = with_human

    def results2json(self, results, outfile_prefix):
        '''save the bbox and attribute result to json file'''
        result_files = dict()
        if len(results[0]) == 2:
            json_results = self._attribute_2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif len(results[0]) == 3:
            json_results = self._segm2json(results)
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results, result_files['segm'])
        return result_files
    
    def _attribute_2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result, attribute_result = results[idx]
            result = result[0]
            for label in range(len(result)):
                bboxes = result[label]
                attributes = attribute_result[label]
                for i in range(bboxes.shape[0]):
                    attribute_ids = attributes[i]
                    attribute_ids = \
                        np.where(attribute_ids <= 234, attribute_ids, attribute_ids+46)
                    attribute_ids = \
                        np.where(attribute_ids <= 283, attribute_ids, attribute_ids+1)
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['attribute_ids'] = attribute_ids.tolist()
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result, seg, attribute_result = results[idx]
            seg = encode_mask_results(seg)
            result = result[0]
            if self.with_human:
                result = result[:len(self.CLASSES)-1]
                seg = seg[:len(self.CLASSES)-1]
                attribute_result = attribute_result[:len(self.CLASSES)-1]
            for label in range(len(result)):
                bboxes = result[label]
                attributes = attribute_result[label]
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    attribute_ids = attributes[i]
                    attribute_ids = \
                        np.where(attribute_ids <= 234, attribute_ids, attribute_ids+46)
                    attribute_ids = \
                        np.where(attribute_ids <= 283, attribute_ids, attribute_ids+1)
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['attribute_ids'] = attribute_ids.tolist()
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    json_results.append(data)
        return json_results
