import pandas as pd
import numpy as np

from ocr import OCR
from postprocess import Postprocess
from utils import DataLoader

DATA_DIR = "./data"
IMAGE_DIR = "./images"
RESULT_DIR = "./results"

ocr = OCR()
postprocess = Postprocess()
loader = DataLoader(data_path = DATA_DIR, image_path = IMAGE_DIR)

data, images = loader.load_data('SAMPLE')

data['output'] = data['image_link'].apply(lambda x: os.path.join(images, x.split('/')[-1])).apply(lambda x: ocr.find_text(x))

data = postprocess.process_units(data)

data.to_csv(os.path.join(RESULT_DIR, 'submission.csv'), index = False)
