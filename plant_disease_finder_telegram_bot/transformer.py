import numpy as np
import albumentations as A
from fastai.vision.core import PILImage
from fastai import *
# from fastcore.transform import Transform


from fastai.vision.all import *

# aug = A.CoarseDropout(p=1, min_holes=40, max_holes=50)


class MyTransform(Transform):
    split_idx = None  # runs on training and valid (0 for train, 1 for valid)
    order = 2  # runs after initial resize
    aug = A.CoarseDropout(p=1, min_holes=40, max_holes=50)

    def __init__(self, aug):
        super().__init__()
        self.aug = aug

    def encodes(self, img: PILImage):
        aug_img = self.aug(image=np.array(img))['image']
        return PILImage.create(aug_img)
