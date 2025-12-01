"""
CheXpert Graph Conditioning Dataset Loader

Aligns three data sources for ControlNet training:
1. CheXpert chest X-ray images
2. Graph embeddings from medical reports
3. Text prompts (clinical impressions/findings)

Direct 1:1 mapping: CSV row index → HDF5 embedding index
"""

import os
from typing import Dict, Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import h5py
from torchvision import transforms


class CheXpertGraphDataset(Dataset):
    """
    Dataset that loads CheXpert images with corresponding graph embeddings.
    
    Args:
        csv_path: Path to reports_processed.csv (223,462 samples)
        embeddings_path: Path to new_embeddings_expanded.h5 (223,462 embeddings)
        image_root: Root directory for CheXpert images (/mnt/data/CheXpert/PNG)
        split: "train" or "valid" to filter by path_to_image prefix
        image_size: Target image size for resizing (default: 512 for RadEdit)
        text_field: Column to use for text prompts ("section_impression" or "section_findings")
        max_samples: Optional limit on dataset size (for testing)
    """
    
    def __init__(
        self,
        csv_path: str = "/home/luab/graph/reports_processed.csv",
        embeddings_path: str = "/home/luab/graph/new_embeddings_expanded.h5",
        image_root: str = "/mnt/data/CheXpert/PNG",
        split: str = "train",
        image_size: int = 512,
        text_field: str = "section_impression",
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        
        print(f"Loading CheXpertGraphDataset (split={split})...")
        
        # Load CSV metadata
        self.df = pd.read_csv(csv_path)
        print(f"  Loaded CSV: {len(self.df)} total samples")
        
        # Filter by split (train/valid based on path_to_image prefix)
        self.df = self.df[self.df['path_to_image'].str.startswith(f"{split}/")]
        print(f"  After {split} filter: {len(self.df)} samples")
        
        # Reset index but keep original index as column for HDF5 lookup
        self.df = self.df.reset_index(drop=False)
        self.df = self.df.rename(columns={'index': 'original_idx'})
        
        # Limit samples if specified (for testing)
        if max_samples is not None:
            self.df = self.df.iloc[:max_samples]
            print(f"  Limited to {max_samples} samples for testing")
        
        # Open HDF5 file (keep handle open for efficient random access)
        print(f"  Opening HDF5 embeddings: {embeddings_path}")
        self.embeddings_file = h5py.File(embeddings_path, 'r')
        self.embeddings = self.embeddings_file['embeddings']
        print(f"  HDF5 shape: {self.embeddings.shape}")
        
        # Store configuration
        self.image_root = image_root
        self.text_field = text_field
        self.image_size = image_size
        self.split = split
        
        # Image transforms for RadEdit compatibility
        # RadEdit expects: [3, 512, 512] normalized to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1, 1]
        ])
        
        print(f"✓ Dataset ready: {len(self)} samples")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a sample with image, graph embedding, and text prompt.
        
        Returns:
            dict: {
                'image': torch.Tensor [3, H, W] in [-1, 1],
                'graph_embedding': torch.Tensor [128, 768],
                'text_prompt': str,
                'patient_id': str,
                'image_path': str,
                'original_idx': int (for debugging HDF5 alignment)
            }
        """
        row = self.df.iloc[idx]
        original_idx = int(row['original_idx'])  # Original CSV index for HDF5
        
        # 1. Load image
        # Note: CSV has .jpg extension but actual files are .png
        img_path = os.path.join(self.image_root, row['path_to_image'])
        img_path = img_path.replace('.jpg', '.png')  # Fix extension mismatch
        
        try:
            # Load as PIL Image, convert to RGB
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            # Return black image as fallback
            image_tensor = torch.zeros(3, self.image_size, self.image_size)
        
        # 2. Load graph embedding from HDF5
        graph_embedding = torch.from_numpy(
            self.embeddings[original_idx].astype(np.float32)  # [128, 768]
        )
        
        # 3. Extract text prompt (handle NaN/empty values)
        text_prompt = row[self.text_field]
        if pd.isna(text_prompt) or text_prompt == "":
            # Fallback to other text fields
            if self.text_field == "section_impression" and 'section_findings' in row:
                text_prompt = row['section_findings']
            if pd.isna(text_prompt) or text_prompt == "":
                text_prompt = "Chest X-ray"  # Final fallback
        
        # Clean text prompt (remove excessive whitespace, newlines)
        text_prompt = ' '.join(str(text_prompt).split())
        
        return {
            'image': image_tensor,  # [3, 512, 512] in [-1, 1]
            'graph_embedding': graph_embedding,  # [128, 768]
            'text_prompt': text_prompt,  # str
            'patient_id': row['deid_patient_id'],
            'image_path': row['path_to_image'],
            'original_idx': original_idx,
        }
    
    def __del__(self):
        """Close HDF5 file handle on cleanup."""
        if hasattr(self, 'embeddings_file'):
            self.embeddings_file.close()


def get_dataloaders(
    batch_size: int = 4,
    num_workers: int = 4,
    csv_path: str = "/home/luab/graph/reports_processed.csv",
    embeddings_path: str = "/home/luab/graph/new_embeddings_expanded.h5",
    image_root: str = "/mnt/data/CheXpert/PNG",
    image_size: int = 512,
    text_field: str = "section_impression",
    max_samples: Optional[int] = None,
):
    """
    Create train and validation dataloaders.
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        csv_path: Path to reports CSV
        embeddings_path: Path to HDF5 embeddings
        image_root: Root directory for images
        image_size: Target image size
        text_field: Text field to use for prompts
        max_samples: Optional limit on dataset size (for testing)
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = CheXpertGraphDataset(
        csv_path=csv_path,
        embeddings_path=embeddings_path,
        image_root=image_root,
        split="train",
        image_size=image_size,
        text_field=text_field,
        max_samples=max_samples,
    )
    
    val_dataset = CheXpertGraphDataset(
        csv_path=csv_path,
        embeddings_path=embeddings_path,
        image_root=image_root,
        split="valid",
        image_size=image_size,
        text_field=text_field,
        max_samples=max_samples // 10 if max_samples else None,  # 10% for val
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # For stable batch size in training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    """Test dataset loading."""
    print("=" * 80)
    print("Testing CheXpertGraphDataset")
    print("=" * 80)
    
    # Create dataset with small sample for testing
    dataset = CheXpertGraphDataset(
        split="train",
        max_samples=10,
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Test loading first sample
    print("\nLoading first sample...")
    sample = dataset[0]
    
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    print(f"  Graph embedding shape: {sample['graph_embedding'].shape}")
    print(f"  Text prompt: {sample['text_prompt'][:100]}...")
    print(f"  Patient ID: {sample['patient_id']}")
    print(f"  Image path: {sample['image_path']}")
    print(f"  Original index: {sample['original_idx']}")
    
    # Test batch loading
    print("\nTesting DataLoader...")
    from torch.utils.data import DataLoader
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    
    print(f"  Batch image shape: {batch['image'].shape}")
    print(f"  Batch graph embedding shape: {batch['graph_embedding'].shape}")
    print(f"  Batch text prompts: {len(batch['text_prompt'])} prompts")
    
    print("\n✓ Dataset test passed!")

