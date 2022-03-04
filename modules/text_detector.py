import time

import numpy as np
import cv2
import onnxruntime as ort
import pyclipper


def polygon_area(points):
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2.0
    return area


def polygon_perimeter(points):
    perimeter = 0
    for i in range(len(points) - 1):
        pt1 = points[i]
        pt2 = points[i + 1]
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
        perimeter += dist
    # final distance to close polygon
    pt1 = points[-1]
    pt2 = points[0]
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    perimeter += dist
    return perimeter


def resize_image(img, limit_side_len=960):
    h, w, _ = img.shape
    # limit the max side
    if max(h, w) > limit_side_len:
        if h > w:
            ratio = float(limit_side_len) / h
        else:
            ratio = float(limit_side_len) / w
    else:
        ratio = 1.0
    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)
    img = cv2.resize(img, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return img, [ratio_h, ratio_w]


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def box_score_slow(bitmap, contour):
    '''
    box_score_slow: use polyon mean score as the mean score
    '''
    h, w = bitmap.shape[:2]
    contour = contour.copy()
    contour = np.reshape(contour, (-1, 2))

    xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
    xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
    ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
    ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    contour[:, 0] = contour[:, 0] - xmin
    contour[:, 1] = contour[:, 1] - ymin
    cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]


def unclip(box, unclip_ratio: float):
    area = polygon_area(box)
    length = polygon_perimeter(box)
    distance = area * unclip_ratio / length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def boxes_from_bitmap(pred, _bitmap, dest_width, dest_height, box_thresh=0.5, unclip_ratio=2.0, max_candidates=1000):
    '''
    _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
    '''

    bitmap = _bitmap
    height, width = bitmap.shape
    min_size = 3

    outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = outs[0], outs[1]
    num_contours = min(len(contours), max_candidates)

    boxes = []
    scores = []
    for index in range(num_contours):
        contour = contours[index]
        points, sside = get_mini_boxes(contour)
        if sside < min_size:
            continue
        points = np.array(points)
        score = box_score_slow(pred, contour)
        if box_thresh > score:
            continue

        box = unclip(points, unclip_ratio=unclip_ratio).reshape(-1, 1, 2)
        box, sside = get_mini_boxes(box)
        if sside < min_size + 2:
            continue
        box = np.array(box)

        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.astype(np.int16))
        scores.append(score)
    return np.array(boxes, dtype=np.int16), scores


def differentiable_binarize(
    pred: np.ndarray, shape_list: np.ndarray, thresh=0.3, box_thresh=0.5, max_candidates=2000, unclip_ratio=2.0
):
    pred = pred[:, 0, :, :]
    segmentation = pred > thresh

    boxes_batch = []
    for batch_index in range(pred.shape[0]):
        src_h, src_w, _, _ = shape_list[batch_index]
        mask = segmentation[batch_index]

        boxes, _ = boxes_from_bitmap(
            pred[batch_index],
            mask,
            src_w,
            src_h,
            box_thresh=box_thresh,
            unclip_ratio=unclip_ratio,
            max_candidates=max_candidates,
        )

        boxes_batch.append({'points': boxes})
    return boxes_batch


class TextDetector:
    def __init__(self, onnx_model_path: str, thresh=0.3, box_thresh=0.5, unclip_ratio=2.0):
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.unclip_ratio = unclip_ratio

        self.onnx_model_path = onnx_model_path
        self.predictor = ort.InferenceSession(
            onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_tensor_name = self.predictor.get_inputs()[0].name

    def _order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def _clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def _filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self._order_points_clockwise(box)
            box = self._clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img: np.ndarray):
        st = time.time()
        ori_im: np.ndarray = img.copy()

        # resize
        src_h, src_w, _ = img.shape
        img, [ratio_h, ratio_w] = resize_image(img)
        shape_list = np.array([src_h, src_w, ratio_h, ratio_w])

        # normalize
        shape = (1, 1, 3)
        scale = np.float32(1.0 / 255.0)
        mean = np.array([0.229, 0.224, 0.225]).reshape(shape).astype('float32')
        std = np.array([0.485, 0.456, 0.406]).reshape(shape).astype('float32')
        img = (img.astype('float32') * scale - mean) / std

        # to chw
        img = img.transpose((2, 0, 1))

        # run inference
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        input_dict = {self.input_tensor_name: img}
        outputs = self.predictor.run(None, input_dict)

        # differentiable binarize
        preds = outputs[0]
        post_result = differentiable_binarize(
            preds, shape_list, thresh=self.thresh, box_thresh=self.box_thresh, unclip_ratio=self.unclip_ratio
        )
        dt_boxes = post_result[0]['points']
        dt_boxes = self._filter_tag_det_res(dt_boxes, ori_im.shape)

        return dt_boxes, time.time() - st
