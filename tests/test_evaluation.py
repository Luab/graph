"""
Pytest tests for the evaluation module.

Test suite covering:
- Single image pair evaluation
- Batch evaluation
- Classifier metrics
- Image metrics
- Available pathologies
"""

import pytest
import logging
from PIL import Image
import torch

logger = logging.getLogger(__name__)


class TestEditEvaluator:
    """Tests for the main EditEvaluator class."""
    
    def test_evaluator_initialization(self, evaluator):
        """Test that evaluator initializes correctly."""
        assert evaluator is not None
        assert hasattr(evaluator, 'classifier_metrics')
        assert hasattr(evaluator, 'image_metrics')
        assert len(evaluator.pathologies) > 0
    
    def test_single_pair_evaluation(self, evaluator, dummy_image_pair, target_pathology):
        """Test evaluation on a single image pair."""
        image_before, image_after = dummy_image_pair
        
        results = evaluator.evaluate_pair(
            image_before=image_before,
            image_after=image_after,
            target_pathology=target_pathology,
            direction='increase',
            compute_image_metrics=True,
            image_metrics_list=['lpips', 'ssim', 'l1'],
        )
        
        # Verify structure
        assert 'classifier' in results
        assert 'image' in results
        
        # Check classifier results
        assert 'predictions_before' in results['classifier']
        assert 'predictions_after' in results['classifier']
        assert 'target_metrics' in results['classifier']
        
        # Check target metrics
        tm = results['classifier']['target_metrics']
        assert tm['pathology'] == target_pathology
        assert 'flip_rate' in tm
        assert 'mean_confidence_delta' in tm
        assert isinstance(tm['flip_rate'], float)
        assert 0 <= tm['flip_rate'] <= 1
        
        # Check image metrics
        assert 'lpips' in results['image']
        assert 'ssim' in results['image']
        assert 'l1' in results['image']
        assert isinstance(results['image']['lpips'], float)
        
        logger.info(f"✓ Single pair evaluation passed")
        logger.info(f"  Target flip rate: {tm['flip_rate']:.2%}")
        logger.info(f"  LPIPS: {results['image']['lpips']:.4f}")
    
    def test_batch_evaluation(self, evaluator, target_pathology):
        """Test evaluation on a batch of image pairs."""
        batch_size = 8
        images_before = [
            Image.new('L', (512, 512), color=100 + i * 5) 
            for i in range(batch_size)
        ]
        images_after = [
            Image.new('L', (512, 512), color=110 + i * 5) 
            for i in range(batch_size)
        ]
        
        results = evaluator.evaluate_batch(
            images_before=images_before,
            images_after=images_after,
            target_pathology=target_pathology,
            direction='increase',
            compute_image_metrics=True,
            image_metrics_list=['lpips', 'ssim', 'l1'],
        )
        
        # Verify structure
        assert 'batch_size' in results
        assert results['batch_size'] == batch_size
        assert 'classifier' in results
        assert 'image' in results
        
        # Check aggregated classifier results
        assert 'aggregated' in results['classifier']
        assert 'per_image' in results['classifier']
        assert len(results['classifier']['per_image']) == batch_size
        
        agg = results['classifier']['aggregated']
        assert 'target_metrics' in agg
        assert 'per_pathology' in agg
        
        # Check target metrics
        if agg.get('target_metrics'):
            tm = agg['target_metrics']
            assert tm['pathology'] == target_pathology
            assert 0 <= tm['flip_rate'] <= 1
            assert 0 <= tm['intended_flip_rate'] <= 1
        
        # Check non-target preservation
        if agg.get('non_target_preservation'):
            ntp = agg['non_target_preservation']
            assert 0 <= ntp['mean_rate'] <= 1
            assert 0 <= ntp['perfect_preservation_rate'] <= 1
        
        # Check image metrics
        assert 'aggregated' in results['image']
        assert 'per_image' in results['image']
        assert len(results['image']['per_image']) == batch_size
        
        img_agg = results['image']['aggregated']
        assert 'lpips' in img_agg
        assert 'mean' in img_agg['lpips']
        assert 'std' in img_agg['lpips']
        
        logger.info(f"✓ Batch evaluation passed")
        if agg.get('target_metrics'):
            logger.info(f"  Target flip rate: {agg['target_metrics']['flip_rate']:.2%}")
        if img_agg.get('lpips'):
            logger.info(f"  Mean LPIPS: {img_agg['lpips']['mean']:.4f}")
    
    def test_available_pathologies(self, evaluator):
        """Test that pathologies are available and correct."""
        pathologies = evaluator.pathologies
        
        assert isinstance(pathologies, list)
        assert len(pathologies) > 0
        
        # Check for some expected pathologies
        expected_pathologies = [
            'Cardiomegaly',
            'Pneumonia',
            'Edema',
            'Effusion',
            'Atelectasis',
        ]
        
        for expected in expected_pathologies:
            assert expected in pathologies, f"Expected pathology '{expected}' not found"
        
        logger.info(f"✓ Available pathologies test passed")
        logger.info(f"  Total pathologies: {len(pathologies)}")
    
    def test_evaluate_without_target(self, evaluator, dummy_image_pair):
        """Test evaluation without specifying target pathology."""
        image_before, image_after = dummy_image_pair
        
        results = evaluator.evaluate_pair(
            image_before=image_before,
            image_after=image_after,
            target_pathology=None,
            compute_image_metrics=True,
        )
        
        # Should still work, just without target-specific metrics
        assert 'classifier' in results
        assert 'image' in results
        assert 'predictions_before' in results['classifier']
        assert 'predictions_after' in results['classifier']
        
        logger.info(f"✓ Evaluation without target passed")
    
    def test_mismatched_batch_sizes(self, evaluator):
        """Test that mismatched batch sizes raise an error."""
        images_before = [Image.new('L', (512, 512)) for _ in range(5)]
        images_after = [Image.new('L', (512, 512)) for _ in range(3)]
        
        with pytest.raises(AssertionError):
            evaluator.evaluate_batch(
                images_before=images_before,
                images_after=images_after,
            )
        
        logger.info(f"✓ Mismatched batch sizes error handling passed")


class TestClassifierMetrics:
    """Tests for ClassifierMetrics class."""
    
    def test_classifier_initialization(self, device):
        """Test classifier initialization."""
        from evaluation import ClassifierMetrics
        
        classifier = ClassifierMetrics(
            model_name='densenet121-res224-all',
            device=device,
            confidence_threshold=0.1,
        )
        
        assert classifier is not None
        assert classifier.model is not None
        assert len(classifier.pathologies) > 0
        
        logger.info(f"✓ Classifier initialization passed")
    
    def test_predict(self, evaluator, dummy_image):
        """Test prediction on images."""
        predictions = evaluator.classifier_metrics.predict([dummy_image])
        
        assert predictions is not None
        assert predictions.shape[0] == 1  # batch size
        assert predictions.shape[1] == len(evaluator.pathologies)
        
        # Check predictions are in valid range [0, 1]
        assert (predictions >= 0).all()
        assert (predictions <= 1).all()
        
        logger.info(f"✓ Prediction test passed")
        logger.info(f"  Predictions shape: {predictions.shape}")
    
    def test_compute_flip_rate(self, evaluator, dummy_image, target_pathology):
        """Test flip rate computation."""
        import torch
        
        # Create dummy predictions
        batch_size = 4
        num_pathologies = len(evaluator.pathologies)
        
        preds_before = torch.rand(batch_size, num_pathologies)
        preds_after = torch.rand(batch_size, num_pathologies)
        
        results = evaluator.classifier_metrics.compute_flip_rate(
            preds_before,
            preds_after,
            target_pathology=target_pathology,
            direction='increase',
        )
        
        assert 'total_images' in results
        assert results['total_images'] == batch_size
        assert 'per_pathology' in results
        assert 'target_metrics' in results
        
        logger.info(f"✓ Flip rate computation passed")


class TestImageMetrics:
    """Tests for ImageMetrics class."""
    
    def test_image_metrics_initialization(self, device):
        """Test image metrics initialization."""
        from evaluation import ImageMetrics
        
        img_metrics = ImageMetrics(device=device, lpips_net='alex')
        
        assert img_metrics is not None
        assert img_metrics.device == device
        
        logger.info(f"✓ Image metrics initialization passed")
    
    def test_compute_lpips(self, evaluator, dummy_image_pair):
        """Test LPIPS computation."""
        image1, image2 = dummy_image_pair
        
        lpips_value = evaluator.image_metrics.compute_lpips(image1, image2)
        
        assert isinstance(lpips_value, float)
        assert lpips_value >= 0
        
        logger.info(f"✓ LPIPS computation passed")
        logger.info(f"  LPIPS value: {lpips_value:.4f}")
    
    def test_compute_ssim(self, evaluator, dummy_image_pair):
        """Test SSIM computation."""
        image1, image2 = dummy_image_pair
        
        ssim_value = evaluator.image_metrics.compute_ssim(image1, image2)
        
        assert isinstance(ssim_value, float)
        assert -1 <= ssim_value <= 1
        
        logger.info(f"✓ SSIM computation passed")
        logger.info(f"  SSIM value: {ssim_value:.4f}")
    
    def test_compute_psnr(self, evaluator, dummy_image_pair):
        """Test PSNR computation."""
        image1, image2 = dummy_image_pair
        
        psnr_value = evaluator.image_metrics.compute_psnr(image1, image2)
        
        assert isinstance(psnr_value, float)
        assert psnr_value > 0
        
        logger.info(f"✓ PSNR computation passed")
        logger.info(f"  PSNR value: {psnr_value:.2f} dB")
    
    def test_compute_l1(self, evaluator, dummy_image_pair):
        """Test L1 distance computation."""
        image1, image2 = dummy_image_pair
        
        l1_value = evaluator.image_metrics.compute_l1(image1, image2)
        
        assert isinstance(l1_value, float)
        assert l1_value >= 0
        
        logger.info(f"✓ L1 computation passed")
        logger.info(f"  L1 value: {l1_value:.4f}")
    
    def test_compute_l2(self, evaluator, dummy_image_pair):
        """Test L2 distance computation."""
        image1, image2 = dummy_image_pair
        
        l2_value = evaluator.image_metrics.compute_l2(image1, image2)
        
        assert isinstance(l2_value, float)
        assert l2_value >= 0
        
        logger.info(f"✓ L2 computation passed")
        logger.info(f"  L2 value: {l2_value:.4f}")
    
    def test_compute_all_metrics(self, evaluator, dummy_image_pair):
        """Test computing all metrics at once."""
        image1, image2 = dummy_image_pair
        
        metrics = evaluator.image_metrics.compute_all(
            image1,
            image2,
            metrics=['lpips', 'ssim', 'psnr', 'l1', 'l2'],
        )
        
        assert 'lpips' in metrics
        assert 'ssim' in metrics
        assert 'psnr' in metrics
        assert 'l1' in metrics
        assert 'l2' in metrics
        
        for metric_name, value in metrics.items():
            assert isinstance(value, float)
        
        logger.info(f"✓ Compute all metrics passed")


class TestUtilities:
    """Tests for utility functions."""
    
    def test_preprocess_for_xrv(self, dummy_image):
        """Test image preprocessing for TorchXRayVision."""
        from evaluation.utils import preprocess_for_xrv
        
        tensor = preprocess_for_xrv(dummy_image, target_size=224)
        
        assert tensor is not None
        assert tensor.shape[-2:] == (224, 224)
        assert tensor.dtype == torch.float32
        
        logger.info(f"✓ Preprocessing test passed")
        logger.info(f"  Output shape: {tensor.shape}")
    
    def test_batch_images(self):
        """Test batch image processing."""
        from evaluation.utils import batch_images
        
        images = [Image.new('L', (512, 512)) for _ in range(4)]
        batch = batch_images(images, target_size=224, device='cpu')
        
        assert batch.shape == (4, 1, 224, 224)
        
        logger.info(f"✓ Batch images test passed")
        logger.info(f"  Batch shape: {batch.shape}")
    
    def test_compute_statistics(self):
        """Test statistics computation."""
        from evaluation.utils import compute_statistics
        
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        stats = compute_statistics(values)
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats
        assert 'count' in stats
        
        assert stats['count'] == 5
        assert abs(stats['mean'] - 0.3) < 0.01
        
        logger.info(f"✓ Statistics computation passed")


# Integration test
class TestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_full_pipeline(self, evaluator, target_pathology):
        """Test full evaluation pipeline."""
        # Create slightly different images to simulate editing
        batch_size = 4
        images_before = [
            Image.new('L', (512, 512), color=120 + i * 2) 
            for i in range(batch_size)
        ]
        images_after = [
            Image.new('L', (512, 512), color=125 + i * 2) 
            for i in range(batch_size)
        ]
        
        # Run full evaluation
        results = evaluator.evaluate_batch(
            images_before=images_before,
            images_after=images_after,
            target_pathology=target_pathology,
            direction='increase',
            compute_image_metrics=True,
            image_metrics_list=['lpips', 'ssim', 'l1'],
        )
        
        # Verify complete structure
        assert results['batch_size'] == batch_size
        assert 'classifier' in results
        assert 'image' in results
        
        # Verify all expected keys exist
        cls_agg = results['classifier']['aggregated']
        assert 'target_metrics' in cls_agg
        assert 'per_pathology' in cls_agg
        
        img_agg = results['image']['aggregated']
        assert 'lpips' in img_agg
        assert 'ssim' in img_agg
        assert 'l1' in img_agg
        
        # Verify per-image results match batch size
        assert len(results['classifier']['per_image']) == batch_size
        assert len(results['image']['per_image']) == batch_size
        
        logger.info(f"✓ Full pipeline integration test passed")
        logger.info(f"  Evaluated {batch_size} image pairs")
        logger.info(f"  Target: {target_pathology}")
        logger.info(f"  Flip rate: {cls_agg['target_metrics']['flip_rate']:.2%}")
        logger.info(f"  Mean LPIPS: {img_agg['lpips']['mean']:.4f}")


# Mark slow tests
@pytest.mark.slow
class TestSlowOperations:
    """Tests that may take longer to run."""
    
    def test_large_batch(self, evaluator, target_pathology):
        """Test evaluation on a large batch."""
        batch_size = 32
        images_before = [
            Image.new('L', (512, 512), color=100 + i) 
            for i in range(batch_size)
        ]
        images_after = [
            Image.new('L', (512, 512), color=105 + i) 
            for i in range(batch_size)
        ]
        
        results = evaluator.evaluate_batch(
            images_before=images_before,
            images_after=images_after,
            target_pathology=target_pathology,
            compute_image_metrics=True,
        )
        
        assert results['batch_size'] == batch_size
        assert len(results['classifier']['per_image']) == batch_size
        
        logger.info(f"✓ Large batch test passed ({batch_size} images)")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])

