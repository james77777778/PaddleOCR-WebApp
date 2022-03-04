from pathlib import Path
import multiprocessing as mp

import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from modules.paddleocr import PaddleOCR
from modules.utils import draw_results, encode_img


'''
init PaddleOCR model
'''

model = PaddleOCR(
    text_det_onnx_file='models/ch_PP-OCRv2_det_infer.onnx',
    text_cls_onnx_file='models/ch_ppocr_mobile_v2.0_cls_infer.onnx',
    text_rec_onnx_file='models/ch_PP-OCRv2_rec_infer.onnx',
    text_rec_dict_file='models/ppocr_keys_v1.txt',
    det_thresh=0.3,
    det_box_thresh=0.5,
    det_unclip_ratio=2.0,
    cls_thresh=0.9,
    drop_score=0.5,
)


'''
init FastAPI application
'''

app = FastAPI()
base_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=Path(base_dir, 'templates'))
app.mount("/static", StaticFiles(directory=Path(base_dir, "static")), name="static")


@app.get('/')
async def get_web(request: Request, response: Response):
    response.headers["Cache-Control"] = "no-cache, no-store"
    return templates.TemplateResponse("index.html", context={"request": request})


@app.post('/inferences')
async def get_inference(img: UploadFile):
    img_bytes = await img.read()
    img_arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img_arr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    results = model(img_arr)

    img_with_results = draw_results(img_arr, results)
    encoded_img_with_results = encode_img(img_with_results)
    return {'results': results, 'img': encoded_img_with_results}


if __name__ == '__main__':
    mp.freeze_support()
    uvicorn.run(app, host='0.0.0.0', port=9000, reload=False, access_log=False, use_colors=False)
