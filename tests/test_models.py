from typing import List
from src.utils.image_processing import binarize_image
import numpy as np

from src.models import ModelLoader
from PIL import Image
import cv2

model_loader = ModelLoader()


def mol_rec_test(image_path):
    mol_rec = model_loader.mol_rec
    group_seg = model_loader.r_group_seg
    res = mol_rec.predict([cv2.imread(image_path)], r_sym_det=group_seg)
    print(res)