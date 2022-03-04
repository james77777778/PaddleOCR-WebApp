import copy
import logging

from modules.text_detector import TextDetector
from modules.text_classifier import TextClassifier
from modules.text_recognizer import TextRecognizer
from modules.utils import get_sorted_boxes, get_rotate_crop_image


class PaddleOCR:
    def __init__(
        self,
        text_det_onnx_file='models/ch_ppocr_server_v2.0_det_infer.onnx',
        text_cls_onnx_file='models/ch_ppocr_mobile_v2.0_cls_infer.onnx',
        text_rec_onnx_file='models/ch_ppocr_server_v2.0_rec_infer.onnx',
        text_rec_dict_file='models/ppocr_keys_v1.txt',
        det_thresh=0.3,
        det_box_thresh=0.5,
        det_unclip_ratio=2.0,
        cls_thresh=0.9,
        drop_score=0.5,
    ):
        self.det_thresh = det_thresh
        self.det_box_thresh = det_box_thresh
        self.det_unclip_ratio = det_unclip_ratio
        self.cls_thresh = cls_thresh
        self.drop_score = drop_score

        self.text_detector = TextDetector(
            text_det_onnx_file, self.det_thresh, self.det_box_thresh, self.det_unclip_ratio
        )
        self.text_classifier = TextClassifier(text_cls_onnx_file, self.cls_thresh)
        self.text_recognizer = TextRecognizer(text_rec_onnx_file, text_rec_dict_file)

    def __call__(self, img):
        ori_im = img.copy()

        # text detection
        dt_boxes, elapse = self.text_detector(img)
        logging.debug(f"dt_boxes num : {len(dt_boxes)}, elapse : {elapse}")
        if dt_boxes is None:
            return None, None

        # get dt_boxes sorted and crop (with rotation)
        dt_boxes = get_sorted_boxes(dt_boxes)
        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        # text classification
        img_crop_list, angle_list, elapse = self.text_classifier(img_crop_list)
        logging.debug(f"cls num  : {len(img_crop_list)}, elapse : {elapse}")

        # text recognition
        rec_res, elapse = self.text_recognizer(img_crop_list)
        logging.debug(f"rec_res num  : {len(rec_res)}, elapse : {elapse}")

        # drop results with `self.drop_score`
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            _, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)

        return [[box.tolist(), str(res[0]), float(res[1])] for box, res in zip(filter_boxes, filter_rec_res)]
