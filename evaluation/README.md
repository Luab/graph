# Evaluation Module

Comprehensive evaluation module for assessing chest X-ray image editing quality using classifier-based and perceptual metrics.

## Features

- **Classifier-Based Metrics**: Uses [TorchXRayVision](https://github.com/mlmed/torchxrayvision) pre-trained models
  - Target flip rate: % of successful edits on target pathology
  - Non-target preservation: % of unrelated pathologies preserved
  - Confidence deltas: Changes in prediction confidence
  - Unintended flips: Which pathologies changed unintentionally

- **Image Quality Metrics**:
  - LPIPS: Learned perceptual similarity
  - SSIM: Structural similarity index
  - PSNR: Peak signal-to-noise ratio
  - L1/L2: Pixel-level distances

- **Flexible Evaluation**:
  - Single image pair or batch processing
  - Per-image and aggregated statistics
  - Optional Weights & Biases logging

## Installation

```bash
pip install -r evaluation/requirements.txt
```

## Quick Start

```python
from evaluation import EditEvaluator

# Initialize
evaluator = EditEvaluator(
    classifier_model='densenet121-res224-all',  # or 'resnet50-res512-all'
    device='cuda',
    confidence_threshold=0.1,  # minimum change to count as flip
    wandb_log=False,           # set True to log to wandb
)

# Evaluate single pair
results = evaluator.evaluate_pair(
    image_before=original_image,
    image_after=edited_image,
    target_pathology='Cardiomegaly',
    direction='increase',
    compute_image_metrics=True,
)

# Evaluate batch (more efficient)
results = evaluator.evaluate_batch(
    images_before=[img1, img2, ...],
    images_after=[edited1, edited2, ...],
    target_pathology='Cardiomegaly',
    direction='increase',
)
```

## Architecture

```
evaluation/
├── __init__.py           # Public API
├── evaluator.py          # Main EditEvaluator class
├── classifier_metrics.py # TorchXRayVision integration
├── image_metrics.py      # LPIPS, SSIM, PSNR, etc.
└── utils.py              # Preprocessing utilities
```

### Design Principles

1. **Modular**: Separate classifier and image metrics for flexibility
2. **Efficient**: Batch processing with minimal redundant computation
3. **KISS**: Clean API, logger + optional wandb, no built-in visualization
4. **Focused**: Evaluates X₁ vs X₂, baseline comparisons handled at experiment level

## API Reference

### EditEvaluator

Main class for comprehensive evaluation.

**Constructor:**
```python
EditEvaluator(
    classifier_model='densenet121-res224-all',  # TorchXRayVision model
    device='cuda',                               # 'cuda' or 'cpu'
    confidence_threshold=0.1,                    # minimum significant change
    lpips_net='alex',                            # 'alex' or 'vgg'
    wandb_log=False,                             # enable wandb logging
)
```

**Methods:**

#### `evaluate_pair()`
Evaluate a single image pair.

```python
results = evaluator.evaluate_pair(
    image_before,                    # PIL Image, torch.Tensor, or numpy array
    image_after,
    target_pathology='Cardiomegaly', # optional
    direction='increase',            # 'increase', 'decrease', or None
    compute_image_metrics=True,
    image_metrics_list=['lpips', 'ssim', 'psnr', 'l1'],
)
```

**Returns:**
```python
{
    'classifier': {
        'target_metrics': {
            'pathology': 'Cardiomegaly',
            'flip_rate': 0.85,
            'intended_flip_rate': 0.85,
            'mean_confidence_delta': 0.42,
        },
        'per_pathology': {
            'Cardiomegaly': {...},
            'Edema': {...},
            ...
        },
        'predictions_before': {...},
        'predictions_after': {...},
    },
    'image': {
        'lpips': 0.123,
        'ssim': 0.856,
        'psnr': 28.4,
        'l1': 0.042,
    }
}
```

#### `evaluate_batch()`
Evaluate multiple image pairs efficiently.

```python
results = evaluator.evaluate_batch(
    images_before,          # list of images
    images_after,           # list of images (same length)
    target_pathology='Cardiomegaly',
    direction='increase',
    compute_image_metrics=True,
    image_metrics_list=['lpips', 'ssim', 'l1'],
)
```

**Returns:**
```python
{
    'batch_size': 32,
    'target_pathology': 'Cardiomegaly',
    'classifier': {
        'aggregated': {
            'target_metrics': {...},
            'non_target_preservation': {
                'mean_rate': 0.92,
                'perfect_preservation_rate': 0.65,
            },
            'unintended_flips': [
                {'pathology': 'Edema', 'flip_rate': 0.15},
                {'pathology': 'Effusion', 'flip_rate': 0.08},
            ],
            'per_pathology': {...},
        },
        'per_image': [
            {'image_idx': 0, 'predictions_before': {...}, ...},
            ...
        ]
    },
    'image': {
        'aggregated': {
            'lpips': {'mean': 0.123, 'std': 0.045, ...},
            'ssim': {'mean': 0.856, 'std': 0.082, ...},
        },
        'per_image': [
            {'image_idx': 0, 'lpips': 0.134, 'ssim': 0.842, ...},
            ...
        ]
    }
}
```

### ClassifierMetrics

Direct access to classifier-based metrics.

```python
from evaluation import ClassifierMetrics

classifier = ClassifierMetrics(
    model_name='densenet121-res224-all',
    device='cuda',
    confidence_threshold=0.1,
)

# Get predictions
predictions = classifier.predict([image1, image2, ...])

# Compute flip rates
results = classifier.compute_flip_rate(
    preds_before, 
    preds_after,
    target_pathology='Cardiomegaly',
)
```

### ImageMetrics

Direct access to image quality metrics.

```python
from evaluation import ImageMetrics

img_metrics = ImageMetrics(device='cuda', lpips_net='alex')

# Individual metrics
lpips = img_metrics.compute_lpips(image1, image2)
ssim = img_metrics.compute_ssim(image1, image2)

# All at once
metrics = img_metrics.compute_all(
    image1, 
    image2,
    metrics=['lpips', 'ssim', 'psnr', 'l1', 'l2']
)
```

## Supported Pathologies

The DenseNet-121 model classifies 18 pathologies:

1. Atelectasis
2. Consolidation
3. Infiltration
4. Pneumothorax
5. Edema
6. Emphysema
7. Fibrosis
8. Effusion
9. Pneumonia
10. Pleural_Thickening
11. Cardiomegaly
12. Nodule
13. Mass
14. Hernia
15. Lung Lesion
16. Fracture
17. Lung Opacity
18. Enlarged Cardiomediastinum

## Metrics Interpretation

### Classifier Metrics

| Metric | Range | Better | Description |
|--------|-------|--------|-------------|
| Target Flip Rate | [0, 1] | Higher | % images where target changed |
| Intended Flip Rate | [0, 1] | Higher | % images where target changed in intended direction |
| Confidence Delta | [-1, 1] | Depends | Mean change in target confidence |
| Non-target Preservation | [0, 1] | Higher | % of non-target pathologies preserved |
| Perfect Preservation Rate | [0, 1] | Higher | % images with zero non-target changes |

### Image Metrics

| Metric | Range | Better | Description |
|--------|-------|--------|-------------|
| LPIPS | [0, ∞) | Lower | Perceptual similarity (learned) |
| SSIM | [0, 1] | Higher | Structural similarity |
| PSNR | [0, ∞) | Higher | Peak signal-to-noise ratio (dB) |
| L1 | [0, ∞) | Lower | Mean absolute pixel error |
| L2 (MSE) | [0, ∞) | Lower | Mean squared pixel error |

## Examples

### Example 1: Basic Usage

```python
from evaluation import EditEvaluator
from PIL import Image

evaluator = EditEvaluator()

original = Image.open('chest_xray.jpg')
edited = Image.open('chest_xray_edited.jpg')

results = evaluator.evaluate_pair(
    original, 
    edited,
    target_pathology='Cardiomegaly',
    direction='increase',
)

print(f"Target flip rate: {results['classifier']['target_metrics']['flip_rate']:.2%}")
print(f"LPIPS: {results['image']['lpips']:.4f}")
```

### Example 2: Batch Evaluation with Wandb

```python
import wandb
from evaluation import EditEvaluator

# Initialize wandb
wandb.init(project='cxr-editing', name='experiment-1')

# Initialize evaluator with wandb logging
evaluator = EditEvaluator(wandb_log=True)

# Load your images
originals = [...]  # list of images
edits = [...]      # list of edited images

# Evaluate - results automatically logged to wandb
results = evaluator.evaluate_batch(
    originals,
    edits,
    target_pathology='Cardiomegaly',
    direction='increase',
)

wandb.finish()
```

### Example 3: Custom Metrics Selection

```python
evaluator = EditEvaluator()

# Only compute specific metrics for speed
results = evaluator.evaluate_batch(
    images_before,
    images_after,
    target_pathology='Pneumonia',
    compute_image_metrics=True,
    image_metrics_list=['lpips', 'l1'],  # skip SSIM, PSNR
)
```

## Testing

The module includes a comprehensive pytest test suite.

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage report
pytest --cov=evaluation --cov-report=html

# Run specific test class
pytest tests/test_evaluation.py::TestEditEvaluator -v

# Run excluding slow tests
pytest -m "not slow"

# Quick test script
./run_tests.sh
```

See [tests/README.md](../tests/README.md) for detailed testing documentation.

## Performance Tips

1. **Batch Processing**: Use `evaluate_batch()` instead of looping `evaluate_pair()` for 10-100x speedup
2. **Metric Selection**: Only compute metrics you need (LPIPS is slowest)
3. **Device**: Use CUDA if available for significant speedup
4. **Model Choice**: DenseNet-224 is faster than ResNet-512 with similar accuracy

## References

- [TorchXRayVision Paper](https://arxiv.org/abs/2111.00595)
- [TorchXRayVision GitHub](https://github.com/mlmed/torchxrayvision)
- [LPIPS Paper](https://arxiv.org/abs/1801.03924)

## License

See parent project license.

