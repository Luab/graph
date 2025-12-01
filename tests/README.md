# Tests

Pytest test suite for the evaluation module.

## Running Tests

### Run all tests
```bash
pytest
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test class
```bash
pytest tests/test_evaluation.py::TestEditEvaluator
```

### Run specific test
```bash
pytest tests/test_evaluation.py::TestEditEvaluator::test_single_pair_evaluation
```

### Run tests excluding slow ones
```bash
pytest -m "not slow"
```

### Run only integration tests
```bash
pytest -m integration
```

### Run with output capture disabled (see prints)
```bash
pytest -s
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py           # Shared fixtures and configuration
├── test_evaluation.py    # Main evaluation module tests
└── README.md            # This file
```

## Test Coverage

### TestEditEvaluator
- Evaluator initialization
- Single image pair evaluation
- Batch evaluation
- Available pathologies
- Evaluation without target pathology
- Error handling (mismatched batch sizes)

### TestClassifierMetrics
- Classifier initialization
- Predictions on images
- Flip rate computation

### TestImageMetrics
- Image metrics initialization
- LPIPS computation
- SSIM computation
- PSNR computation
- L1 distance computation
- L2 distance computation
- Computing all metrics at once

### TestUtilities
- Image preprocessing for TorchXRayVision
- Batch image processing
- Statistics computation

### TestIntegration
- Full evaluation pipeline

### TestSlowOperations (marked as slow)
- Large batch evaluation

## Fixtures

Available fixtures from `conftest.py`:

- `device`: CUDA if available, else CPU
- `evaluator`: Shared EditEvaluator instance
- `dummy_image`: Single dummy grayscale image
- `dummy_image_pair`: Pair of dummy images
- `dummy_batch`: Batch of dummy image pairs
- `target_pathology`: Default target pathology ('Cardiomegaly')

## Adding New Tests

1. Create test functions starting with `test_`
2. Use fixtures for common setup
3. Group related tests in classes starting with `Test`
4. Mark slow tests with `@pytest.mark.slow`
5. Use descriptive assertions

Example:
```python
def test_my_feature(evaluator, dummy_image_pair):
    """Test my new feature."""
    image1, image2 = dummy_image_pair
    
    result = evaluator.my_feature(image1, image2)
    
    assert result is not None
    assert isinstance(result, dict)
```

## CI/CD Integration

For continuous integration, add to your CI config:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pip install -r evaluation/requirements.txt
    pytest
```

## Coverage

To generate coverage reports (requires pytest-cov):

```bash
pip install pytest-cov
pytest --cov=evaluation --cov-report=html
```

View the report:
```bash
open htmlcov/index.html
```



