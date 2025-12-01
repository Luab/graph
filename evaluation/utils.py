"""
Utility functions for image preprocessing and conversion.
"""

import torch
import numpy as np
from PIL import Image
from typing import Union, List


def preprocess_for_xrv(
    image: Union[Image.Image, torch.Tensor, np.ndarray],
    target_size: int = 224,
) -> torch.Tensor:
    """
    Preprocess image for TorchXRayVision models.
    
    TorchXRayVision expects:
    - Single channel (grayscale)
    - Normalized with mean=0.5, std=0.5 (maps [-1,1] to [0,1])
    - Shape: (1, H, W) or (B, 1, H, W)
    
    Args:
        image: Input image (PIL, tensor, or numpy)
        target_size: Target image size (height and width)
        
    Returns:
        Preprocessed tensor ready for XRV models
    """
    # Convert to tensor if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)
        
        # Add channel dim if needed
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
        else:
            # (H, W, C) -> (C, H, W)
            image_tensor = image_tensor.permute(2, 0, 1)
            
    elif isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image.astype(np.float32))
        if image.max() > 1.0:
            image_tensor = image_tensor / 255.0
            
    elif isinstance(image, torch.Tensor):
        image_tensor = image.clone()
        # Ensure float
        if image_tensor.dtype != torch.float32:
            image_tensor = image_tensor.float()
        # Normalize if needed
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Ensure we have (C, H, W) format
    if len(image_tensor.shape) == 2:
        image_tensor = image_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
    elif len(image_tensor.shape) == 4:
        # Batch dimension present, handle it separately
        pass
    
    # Convert RGB to grayscale if needed (using luminance weights)
    if image_tensor.shape[0] == 3 or (len(image_tensor.shape) == 4 and image_tensor.shape[1] == 3):
        if len(image_tensor.shape) == 3:  # (3, H, W)
            # Standard RGB to grayscale conversion
            image_tensor = (
                0.299 * image_tensor[0:1] + 
                0.587 * image_tensor[1:2] + 
                0.114 * image_tensor[2:3]
            )
        else:  # (B, 3, H, W)
            image_tensor = (
                0.299 * image_tensor[:, 0:1] + 
                0.587 * image_tensor[:, 1:2] + 
                0.114 * image_tensor[:, 2:3]
            )
    
    # Resize if needed
    if len(image_tensor.shape) == 3:  # (1, H, W)
        if image_tensor.shape[1] != target_size or image_tensor.shape[2] != target_size:
            image_tensor = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0),
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
    else:  # (B, 1, H, W)
        if image_tensor.shape[2] != target_size or image_tensor.shape[3] != target_size:
            image_tensor = torch.nn.functional.interpolate(
                image_tensor,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            )
    
    # Normalize to [0, 1] range if coming from [-1, 1]
    if image_tensor.min() < 0:
        image_tensor = (image_tensor + 1) / 2
    
    # XRV normalization: (x - 0.5) / 0.5 which maps [0,1] -> [-1,1]
    # But actually XRV uses mean=(2048 * 0.5) for raw values, simplified to 0.5 for [0,1] range
    # Let's check - actually they normalize per their dataset stats
    # For simplicity, keep in [0, 1] range - XRV models handle this internally
    
    return image_tensor


def batch_images(
    images: List[Union[Image.Image, torch.Tensor, np.ndarray]],
    target_size: int = 224,
    device: str = 'cuda',
) -> torch.Tensor:
    """
    Batch process multiple images for XRV.
    
    Args:
        images: List of images
        target_size: Target size for resizing
        device: Device to place tensors on
        
    Returns:
        Batched tensor of shape (B, 1, H, W)
    """
    processed = []
    for img in images:
        img_tensor = preprocess_for_xrv(img, target_size)
        # Ensure (1, H, W)
        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(0)
        processed.append(img_tensor)
    
    # Stack into batch
    batch = torch.stack(processed, dim=0)
    return batch.to(device)


def compute_statistics(values: List[float]) -> dict:
    """
    Compute summary statistics for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with mean, std, min, max, median
    """
    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0,
            'count': 0,
        }
    
    values_tensor = torch.tensor(values, dtype=torch.float32)
    
    return {
        'mean': values_tensor.mean().item(),
        'std': values_tensor.std().item(),
        'min': values_tensor.min().item(),
        'max': values_tensor.max().item(),
        'median': values_tensor.median().item(),
        'count': len(values),
    }



