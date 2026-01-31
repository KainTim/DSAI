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
from scipy.ndimage import gaussian_filter, map_coordinates

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

def preprocess(input_array: np.ndarray):
    input_array = np.asarray(input_array, dtype=np.float32) / 255.0
    return input_array

def elastic_transform(image: np.ndarray, alpha: float = 20, sigma: float = 4) -> np.ndarray:
    """Apply elastic deformation to image array"""
    shape = image.shape[:2]
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    # Apply to each channel
    transformed = np.zeros_like(image)
    for i in range(image.shape[2]):
        transformed[:, :, i] = map_coordinates(image[:, :, i], indices, order=1, mode='reflect').reshape(shape)
    
    return transformed

def add_noise(img_array: np.ndarray, noise_type: str = 'gaussian', strength: float = 0.02) -> np.ndarray:
    """Add various types of noise to image"""
    if noise_type == 'gaussian':
        noise = np.random.normal(0, strength, img_array.shape)
        noisy = img_array + noise
    elif noise_type == 'salt_pepper':
        noisy = img_array.copy()
        # Salt
        num_salt = int(strength * img_array.size * 0.5)
        coords = [np.random.randint(0, i, num_salt) for i in img_array.shape]
        noisy[coords[0], coords[1], :] = 1
        # Pepper
        num_pepper = int(strength * img_array.size * 0.5)
        coords = [np.random.randint(0, i, num_pepper) for i in img_array.shape]
        noisy[coords[0], coords[1], :] = 0
    else:
        noisy = img_array
    
    return np.clip(noisy, 0, 1)

def augment_image(img: Image, strength: float = 0.8) -> Image:
    """Apply comprehensive data augmentation for better generalization"""
    # Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Random vertical flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Random rotation (90, 180, 270 degrees, or small angles)
    if random.random() > 0.5:
        if random.random() > 0.7:
            # Large rotation
            angle = random.choice([90, 180, 270])
            img = img.rotate(angle)
        else:
            # Small rotation for more variation
            angle = random.uniform(-15, 15)
            img = img.rotate(angle, fillcolor=(128, 128, 128))
    
    # More aggressive color augmentation
    if random.random() > 0.3:
        # Brightness
        factor = 1.0 + random.uniform(-0.3, 0.3) * strength
        img = ImageEnhance.Brightness(img).enhance(factor)
    
    if random.random() > 0.3:
        # Contrast
        factor = 1.0 + random.uniform(-0.3, 0.3) * strength
        img = ImageEnhance.Contrast(img).enhance(factor)
    
    if random.random() > 0.3:
        # Saturation
        factor = 1.0 + random.uniform(-0.25, 0.25) * strength
        img = ImageEnhance.Color(img).enhance(factor)
    
    if random.random() > 0.7:
        # Sharpness
        factor = 1.0 + random.uniform(-0.3, 0.5) * strength
        img = ImageEnhance.Sharpness(img).enhance(factor)
    
    # Gaussian blur for robustness
    if random.random() > 0.8:
        radius = random.uniform(0.5, 1.5) * strength
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    # Convert to array for elastic transform and noise
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Elastic deformation
    if random.random() > 0.7:
        alpha = random.uniform(15, 30) * strength
        sigma = random.uniform(3, 5)
        img_array = elastic_transform(img_array, alpha=alpha, sigma=sigma)
    
    # Add noise
    if random.random() > 0.6:
        noise_type = random.choice(['gaussian', 'salt_pepper'])
        noise_strength = random.uniform(0.01, 0.03) * strength
        img_array = add_noise(img_array, noise_type=noise_type, strength=noise_strength)
    
    # Convert back to PIL Image
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    return img

class ImageDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading images from a folder with augmentation support
    """

    def __init__(self, datafolder: str, augment: bool = True, augment_strength: float = 0.8):
        self.imagefiles = sorted(glob.glob(os.path.join(datafolder,"**","*.jpg"),recursive=True))
        self.augment = augment
        self.augment_strength = augment_strength

    def __len__(self):
        return len(self.imagefiles)
        
    def __getitem__(self, idx:int):
        index = int(idx)
        
        image = Image.open(self.imagefiles[index])
        image = resize(image)
        
        # Apply augmentation
        if self.augment:
            image = augment_image(image, self.augment_strength)
        
        image = np.asarray(image)
        image = preprocess(image)
        spacing_x = random.randint(2,6)
        spacing_y = random.randint(2,6)
        offset_x = random.randint(0,8)
        offset_y = random.randint(0,8)
        spacing = (spacing_x, spacing_y)
        offset = (offset_x, offset_y)
        input_array, known_array = create_arrays_from_image(image.copy(), offset, spacing)
        target_image = torch.from_numpy(np.transpose(image, (2,0,1)))
        input_array = torch.from_numpy(input_array)
        known_array = torch.from_numpy(known_array)
        input_array = torch.cat((input_array, known_array), dim=0)
        return input_array, target_image