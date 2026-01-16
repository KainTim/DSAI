"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    datasets.py
"""

import torch
import numpy as np
import random
import glob
import os
from PIL import Image

IMAGE_DIMENSION = 100


def create_arrays_from_image(image_array: np.ndarray, offset: tuple, spacing: tuple) -> tuple[np.ndarray, np.ndarray]:
    image_array, known_array = None, None

    # TODO: Implement the logic to create input and known arrays based on offset and spacing

    return image_array, known_array

def resize(img: Image):
    pass
def preprocess(input_array: np.ndarray):
    pass

class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading images from a folder
    """

    def __init__(self, datafolder: str):
        self.imagefiles = sorted(glob.glob(os.path.join(datafolder,"**","*.jpg"),recursive=True))

    def __len__(self):
        return len(self.imagefiles)
        
    def __getitem__(self, idx:int):
        pass
        
    # TODO: Implement the __init__, __len__, and __getitem__ methods