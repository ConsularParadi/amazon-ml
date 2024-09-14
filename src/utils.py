import os
from typing import Tuple
import pandas as pd

class DataLoader():
    def __init__(self, data_path: str = "./data", image_path: str = "./images"):
        self.DATA_DIR = data_path
        self.IMAGE_DIR = image_path

    def load_data(self, option: str = 'SAMPLE') -> Tuple[pd.DataFrame, str]:
        match option:
            case 'SAMPLE':
                data_dir = os.path.join(self.DATA_DIR, 'sample.csv')
                img_dir = os.path.join(self.IMAGE_DIR, 'sample')
            case 'TEST':
                data_dir = os.path.join(self.DATA_DIR, 'test.csv')
                img_dir = os.path.join(self.IMAGE_DIR, 'test')
            case 'TRAIN':
                data_dir = os.path.join(self.DATA_DIR, 'train.csv')
                img_dir = os.path.join(self.IMAGE_DIR, 'train')
            case _:
                raise ValueError("Invalid option. Please choose from 'SAMPLE', 'TEST', or 'TRAIN'.")

        return pd.read_csv(data_dir), img_dir



