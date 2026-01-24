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
from PIL import Image, ImageEnhance, ImageFilter

IMAGE_DIMENSION = 100


class DataAugmentation:
    """Data augmentation pipeline for improved generalization"""
    
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image: Image.Image) -> Image.Image:
        # Random horizontal flip
        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random vertical flip
        if random.random() < self.p * 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        # Random rotation (90 degree increments)
        if random.random() < self.p * 0.3:
            angle = random.choice([90, 180, 270])
            image = image.rotate(angle)
        
        # Color jittering
        if random.random() < self.p * 0.4:
            # Brightness
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.85, 1.15))
        
        if random.random() < self.p * 0.4:
            # Contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.85, 1.15))
        
        if random.random() < self.p * 0.3:
            # Saturation
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.uniform(0.85, 1.15))
        
        return image


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
def preprocess(input_array: np.ndarray):
    input_array = np.asarray(input_array, dtype=np.float32) / 255.0
    return input_array

class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading images from a folder with augmentation
    """

    def __init__(self, datafolder: str, augment: bool = True):
        self.imagefiles = sorted(glob.glob(os.path.join(datafolder, "**", "*.jpg"), recursive=True))
        self.augment = augment
        self.augmentation = DataAugmentation(p=0.5) if augment else None

    def __len__(self):
        return len(self.imagefiles)
        
    def __getitem__(self, idx: int):
        index = int(idx)
        
        image = Image.open(self.imagefiles[index]).convert('RGB')
        
        # Apply augmentation before resize
        if self.augment and self.augmentation is not None:
            image = self.augmentation(image)
        
        image = resize(image)
        image = np.asarray(image)
        image = preprocess(image)
        
        # More varied spacing for better generalization
        spacing_x = random.randint(2, 8)
        spacing_y = random.randint(2, 8)
        offset_x = random.randint(0, min(spacing_x - 1, 8))
        offset_y = random.randint(0, min(spacing_y - 1, 8))
        spacing = (spacing_x, spacing_y)
        offset = (offset_x, offset_y)
        
        input_array, known_array = create_arrays_from_image(image.copy(), offset, spacing)
        target_image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        input_array = torch.from_numpy(input_array)
        known_array = torch.from_numpy(known_array)
        input_array = torch.cat((input_array, known_array), dim=0)
        
        return input_array, target_image