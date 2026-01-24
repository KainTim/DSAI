"""
    Author: Your Name
    HTL-Grieskirchen 5. Jahrgang, Schuljahr 2025/26
    datasets.py
"""

from torchvision import transforms
import torch
import numpy as np
import random
import glob
import os
from PIL import Image

IMAGE_DIMENSION = 100


def create_arrays_from_image(image_array: np.ndarray, offset: tuple, spacing: tuple) -> tuple[np.ndarray, np.ndarray]:
    image_array = np.transpose(image_array, (2, 0, 1))
    known_array = np.zeros_like(image_array)

    known_array[:, offset[1]::spacing[1], offset[0]::spacing[0]] = 1

    image_array[known_array == 0] = 0
    known_array = known_array[0:1]

    return image_array, known_array

def resize(img: Image):
    resize_transforms = transforms.Compose([
        transforms.Resize((IMAGE_DIMENSION, IMAGE_DIMENSION)),
        transforms.CenterCrop((IMAGE_DIMENSION, IMAGE_DIMENSION))
    ])
    return resize_transforms(img)


def augment_geometric(img: Image.Image) -> Image.Image:
    """Lightweight, label-preserving augmentation (safe for train/val/test splits)."""
    # Horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    # Vertical flip (less frequent)
    if random.random() < 0.2:
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    # 90-degree rotations (no interpolation artifacts)
    r = random.random()
    if r < 0.25:
        img = img.transpose(Image.Transpose.ROTATE_90)
    elif r < 0.5:
        img = img.transpose(Image.Transpose.ROTATE_180)
    elif r < 0.75:
        img = img.transpose(Image.Transpose.ROTATE_270)
    return img
def preprocess(input_array: np.ndarray):
    input_array = np.asarray(input_array, dtype=np.float32) / 255.0
    return input_array

class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading images from a folder
    """

    def __init__(self, datafolder: str):
        self.imagefiles = sorted(glob.glob(os.path.join(datafolder,"**","*.jpg"),recursive=True))

    def __len__(self):
        return len(self.imagefiles)
        
    def __getitem__(self, idx:int):
        index = int(idx)
        
        image = Image.open(self.imagefiles[index]).convert("RGB")
        image = augment_geometric(image)
        image = np.asarray(resize(image))
        image = preprocess(image)

        # Sample a grid-mask similar in density to the challenge testset (~8% known pixels).
        # IMPORTANT: offset ranges must be tied to spacing to avoid accidental distribution shift.
        spacing_x = random.randint(4, 6)
        spacing_y = random.randint(2, 4)
        offset_x = random.randint(0, spacing_x - 1)
        offset_y = random.randint(0, spacing_y - 1)
        spacing = (spacing_x, spacing_y)
        offset = (offset_x, offset_y)
        input_array, known_array = create_arrays_from_image(image.copy(), offset, spacing)
        target_image = torch.from_numpy(np.transpose(image, (2,0,1)))
        input_array = torch.from_numpy(input_array)
        known_array = torch.from_numpy(known_array)
        input_array = torch.cat((input_array, known_array), dim=0)
        return input_array, target_image