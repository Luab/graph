"""
Classifier-based metrics using TorchXRayVision models.
"""

import torch
import logging
from typing import List, Dict, Optional, Union
from PIL import Image
import numpy as np

from .utils import preprocess_for_xrv, batch_images, compute_statistics

logger = logging.getLogger(__name__)


class ClassifierMetrics:
    """
    Compute classifier-based metrics using TorchXRayVision.
    
    Metrics:
    - Target flip rate: % of edits where target pathology prediction changed as intended
    - Non-target preservation rate: % of non-target pathologies that remained unchanged
    - Confidence delta: Change in prediction confidence for each pathology
    - Unintended flips: List of pathologies that flipped unintentionally
    """
    
    # TorchXRayVision pathologies (DenseNet-121 trained on multiple datasets)
    PATHOLOGIES = [
        'Atelectasis',
        'Consolidation',
        'Infiltration',
        'Pneumothorax',
        'Edema',
        'Emphysema',
        'Fibrosis',
        'Effusion',
        'Pneumonia',
        'Pleural_Thickening',
        'Cardiomegaly',
        'Nodule',
        'Mass',
        'Hernia',
        'Lung Lesion',
        'Fracture',
        'Lung Opacity',
        'Enlarged Cardiomediastinum',
    ]
    
    def __init__(
        self,
        model_name: str = 'densenet121-res224-all',
        device: str = 'cuda',
        confidence_threshold: float = 0.1,
        flip_threshold: float = 0.5,
    ):
        """
        Initialize classifier metrics.
        
        Args:
            model_name: TorchXRayVision model name
                - 'densenet121-res224-all' (default, recommended)
                - 'resnet50-res512-all'
            device: Device for inference
            confidence_threshold: Minimum confidence change to count as significant
            flip_threshold: Threshold for binary classification (0.5 standard)
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.flip_threshold = flip_threshold
        self.model_name = model_name
        
        logger.info(f"Loading TorchXRayVision model: {model_name}")
        
        try:
            import torchxrayvision as xrv
        except ImportError:
            raise ImportError(
                "torchxrayvision not installed. Install with: pip install torchxrayvision"
            )
        
        # Load model
        if model_name == 'densenet121-res224-all':
            self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.input_size = 224
        elif model_name == 'resnet50-res512-all':
            self.model = xrv.models.ResNet(weights="resnet50-res512-all")
            self.input_size = 512
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Store pathology names from model
        self.pathologies = self.model.pathologies
        
        logger.info(f"Model loaded successfully. Pathologies: {len(self.pathologies)}")
    
    @torch.no_grad()
    def predict(
        self,
        images: Union[List, torch.Tensor],
    ) -> torch.Tensor:
        """
        Get classifier predictions for images.
        
        Args:
            images: List of images or batched tensor
            
        Returns:
            Predictions tensor of shape (B, num_pathologies)
        """
        if isinstance(images, list):
            batch = batch_images(images, self.input_size, self.device)
        else:
            batch = images.to(self.device)
        
        # Ensure correct shape (B, 1, H, W)
        if len(batch.shape) == 3:
            batch = batch.unsqueeze(0)
        
        predictions = self.model(batch)
        
        # Apply sigmoid to get probabilities
        predictions = torch.sigmoid(predictions)
        
        return predictions
    
    def compute_flip_rate(
        self,
        preds_before: torch.Tensor,
        preds_after: torch.Tensor,
        target_pathology: Optional[str] = None,
        direction: Optional[str] = None,  # 'increase' or 'decrease'
    ) -> Dict:
        """
        Compute flip rates and confidence changes.
        
        Args:
            preds_before: Predictions before edit (B, num_pathologies)
            preds_after: Predictions after edit (B, num_pathologies)
            target_pathology: Target pathology to track (optional)
            direction: Expected direction of change ('increase' or 'decrease')
            
        Returns:
            Dictionary with flip rate statistics
        """
        batch_size = preds_before.shape[0]
        num_pathologies = preds_before.shape[1]
        
        # Compute binary predictions
        binary_before = (preds_before > self.flip_threshold).float()
        binary_after = (preds_after > self.flip_threshold).float()
        
        # Detect flips (any change in binary prediction)
        flips = (binary_before != binary_after).float()  # (B, num_pathologies)
        
        # Compute confidence changes
        confidence_deltas = preds_after - preds_before  # (B, num_pathologies)
        
        # Significant changes (above threshold)
        significant_changes = (torch.abs(confidence_deltas) > self.confidence_threshold).float()
        
        results = {
            'total_images': batch_size,
            'total_pathologies': num_pathologies,
        }
        
        # Per-pathology statistics
        per_pathology = {}
        for i, pathology_name in enumerate(self.pathologies):
            pathology_flips = flips[:, i].sum().item()
            pathology_significant = significant_changes[:, i].sum().item()
            
            per_pathology[pathology_name] = {
                'flip_count': pathology_flips,
                'flip_rate': pathology_flips / batch_size,
                'significant_change_count': pathology_significant,
                'significant_change_rate': pathology_significant / batch_size,
                'mean_confidence_delta': confidence_deltas[:, i].mean().item(),
                'mean_pred_before': preds_before[:, i].mean().item(),
                'mean_pred_after': preds_after[:, i].mean().item(),
            }
        
        results['per_pathology'] = per_pathology
        
        # Target pathology specific metrics
        if target_pathology:
            if target_pathology not in self.pathologies:
                logger.warning(f"Target pathology '{target_pathology}' not in model pathologies")
                results['target_metrics'] = None
            else:
                target_idx = self.pathologies.index(target_pathology)
                
                target_flips = flips[:, target_idx]
                target_deltas = confidence_deltas[:, target_idx]
                
                # Check if flip is in the intended direction
                if direction == 'increase':
                    intended_flips = (target_deltas > self.confidence_threshold).float()
                elif direction == 'decrease':
                    intended_flips = (target_deltas < -self.confidence_threshold).float()
                else:
                    # Any significant change
                    intended_flips = (torch.abs(target_deltas) > self.confidence_threshold).float()
                
                results['target_metrics'] = {
                    'pathology': target_pathology,
                    'direction': direction,
                    'flip_rate': target_flips.mean().item(),
                    'intended_flip_rate': intended_flips.mean().item(),
                    'mean_confidence_delta': target_deltas.mean().item(),
                    'median_confidence_delta': target_deltas.median().item(),
                }
        
        # Non-target preservation (if target specified)
        if target_pathology and target_pathology in self.pathologies:
            target_idx = self.pathologies.index(target_pathology)
            
            # Mask out target pathology
            non_target_mask = torch.ones(num_pathologies, dtype=torch.bool)
            non_target_mask[target_idx] = False
            
            non_target_flips = flips[:, non_target_mask]
            non_target_preserved = (non_target_flips == 0).float()
            
            # Per-image preservation rate
            preservation_per_image = non_target_preserved.mean(dim=1)  # (B,)
            
            results['non_target_preservation'] = {
                'mean_rate': preservation_per_image.mean().item(),
                'median_rate': preservation_per_image.median().item(),
                'perfect_preservation_count': (preservation_per_image == 1.0).sum().item(),
                'perfect_preservation_rate': (preservation_per_image == 1.0).float().mean().item(),
            }
            
            # Identify which pathologies flipped unintentionally
            unintended_flip_counts = non_target_flips.sum(dim=0)
            unintended_pathologies = []
            
            non_target_pathologies = [p for p in self.pathologies if p != target_pathology]
            for i, count in enumerate(unintended_flip_counts):
                if count > 0:
                    unintended_pathologies.append({
                        'pathology': non_target_pathologies[i],
                        'flip_count': count.item(),
                        'flip_rate': (count / batch_size).item(),
                    })
            
            results['unintended_flips'] = sorted(
                unintended_pathologies, 
                key=lambda x: x['flip_rate'], 
                reverse=True
            )
        
        return results
    
    def evaluate_pair(
        self,
        image_before: Union[Image.Image, torch.Tensor, np.ndarray],
        image_after: Union[Image.Image, torch.Tensor, np.ndarray],
        target_pathology: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate a single image pair.
        
        Args:
            image_before: Original image
            image_after: Edited image
            target_pathology: Target pathology for editing
            direction: Expected direction ('increase' or 'decrease')
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions
        pred_before = self.predict([image_before])
        pred_after = self.predict([image_after])
        
        # Compute flip rates
        results = self.compute_flip_rate(
            pred_before,
            pred_after,
            target_pathology=target_pathology,
            direction=direction,
        )
        
        # Add per-image details
        results['predictions_before'] = {
            p: pred_before[0, i].item() 
            for i, p in enumerate(self.pathologies)
        }
        results['predictions_after'] = {
            p: pred_after[0, i].item() 
            for i, p in enumerate(self.pathologies)
        }
        
        return results
    
    def evaluate_batch(
        self,
        images_before: List,
        images_after: List,
        target_pathology: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate a batch of image pairs.
        
        Args:
            images_before: List of original images
            images_after: List of edited images
            target_pathology: Target pathology for editing
            direction: Expected direction ('increase' or 'decrease')
            
        Returns:
            Dictionary with aggregated and per-image metrics
        """
        assert len(images_before) == len(images_after), "Mismatched batch sizes"
        
        logger.info(f"Evaluating batch of {len(images_before)} image pairs")
        
        # Get predictions for all images
        preds_before = self.predict(images_before)
        preds_after = self.predict(images_after)
        
        # Compute aggregated metrics
        aggregated = self.compute_flip_rate(
            preds_before,
            preds_after,
            target_pathology=target_pathology,
            direction=direction,
        )
        
        # Compute per-image metrics
        per_image = []
        for i in range(len(images_before)):
            img_results = {
                'image_idx': i,
                'predictions_before': {
                    p: preds_before[i, j].item() 
                    for j, p in enumerate(self.pathologies)
                },
                'predictions_after': {
                    p: preds_after[i, j].item() 
                    for j, p in enumerate(self.pathologies)
                },
                'confidence_deltas': {
                    p: (preds_after[i, j] - preds_before[i, j]).item()
                    for j, p in enumerate(self.pathologies)
                },
            }
            
            # Target-specific metrics
            if target_pathology and target_pathology in self.pathologies:
                target_idx = self.pathologies.index(target_pathology)
                delta = preds_after[i, target_idx] - preds_before[i, target_idx]
                
                img_results['target_flip'] = abs(delta.item()) > self.confidence_threshold
                img_results['target_delta'] = delta.item()
                
                # Count non-target flips
                non_target_flips = 0
                for j, p in enumerate(self.pathologies):
                    if j != target_idx:
                        if abs(preds_after[i, j] - preds_before[i, j]) > self.confidence_threshold:
                            non_target_flips += 1
                
                img_results['non_target_flips'] = non_target_flips
            
            per_image.append(img_results)
        
        return {
            'aggregated': aggregated,
            'per_image': per_image,
        }



