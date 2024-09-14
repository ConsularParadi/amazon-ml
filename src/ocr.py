import paddleocr
import pandas as pd
import numpy as np

class OCR:

    def __init__(self):
        self.ocr = paddleocr.PaddleOCR(use_angle_cls=True, lang='en', recgonize_gpu=True)

    def find_text(self, image_path: str):
        results = self.ocr.ocr(image, cls=True)
        try:
            text = '\n'.join([line[1][0] for line in result])
        except:
            text = ""
        return text