"""
Pytest configuration and shared fixtures.
"""

import pytest
import torch
from PIL import Image
import logging

# Setup logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


@pytest.fixture(scope="session")
def device():
    """Get available device for tests."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope="session")
def evaluator(device):
    """Create a shared evaluator instance for tests."""
    from evaluation import EditEvaluator
    
    return EditEvaluator(
        classifier_model='densenet121-res224-all',
        device=device,
        confidence_threshold=0.1,
        wandb_log=False,
    )


@pytest.fixture
def dummy_image():
    """Create a dummy grayscale image."""
    return Image.new('L', (512, 512), color=128)


@pytest.fixture
def dummy_image_pair():
    """Create a pair of dummy images for testing."""
    image_before = Image.new('L', (512, 512), color=128)
    image_after = Image.new('L', (512, 512), color=130)
    return image_before, image_after


@pytest.fixture
def dummy_batch(batch_size=8):
    """Create a batch of dummy image pairs."""
    images_before = [
        Image.new('L', (512, 512), color=100 + i * 5) 
        for i in range(batch_size)
    ]
    images_after = [
        Image.new('L', (512, 512), color=110 + i * 5) 
        for i in range(batch_size)
    ]
    return images_before, images_after


@pytest.fixture
def target_pathology():
    """Default target pathology for testing."""
    return 'Cardiomegaly'



