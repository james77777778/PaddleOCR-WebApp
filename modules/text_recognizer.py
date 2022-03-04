import time
import math

import numpy as np
import cv2
import onnxruntime as ort


class CTCLabelDecode:
    def __init__(self, character_dict_path=None):
        self.beg_str = "sos"
        self.end_str = "eos"

        self.character_str = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character_str.append(line)
        # add space char
        self.character_str.append(" ")
        dict_character = list(self.character_str)

        # add special char
        dict_character = ['blank'] + dict_character

        self.character = dict_character

    def decode(self, text_index, text_prob):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = [0]
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                # remove duplicate
                if idx > 0 and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]:
                    continue
                char_list.append(self.character[int(text_index[batch_idx][idx])])
                conf_list.append(text_prob[batch_idx][idx])
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def __call__(self, preds):
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob)
        return text


class TextRecognizer:
    def __init__(self, onnx_model_path: str, character_dict_path: str):
        self.rec_algorithm = "CRNN"
        self.rec_image_shape = [3, 32, 320]
        self.rec_batch_num = 6
        self.postprocess_op = CTCLabelDecode(character_dict_path=character_dict_path)

        self.onnx_model_path = onnx_model_path
        self.predictor = ort.InferenceSession(
            onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_tensor_name = self.predictor.get_inputs()[0].name

    def _resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((32 * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        st = time.time()

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = self._resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            input_dict = {self.input_tensor_name: norm_img_batch}
            outputs = self.predictor.run(None, input_dict)
            preds = outputs[0]
            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        return rec_res, time.time() - st
