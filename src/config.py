
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Image configuration
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 3
NUM_CLASSES = 43

# Training configuration
BATCH_SIZE = 64
RANDOM_SEED = 42

# Recommended deployment model
RECOMMENDED_MODEL_NAME = "Baseline CNN"
RECOMMENDED_MODEL_PATH = MODELS_DIR / "cnn_baseline.keras"

# Dataset metadata
TRAIN_METADATA = PROCESSED_DATA_DIR / "train_metadata.csv"
VALIDATION_METADATA = PROCESSED_DATA_DIR / "validation_metadata.csv"
TEST_METADATA = PROCESSED_DATA_DIR / "test_metadata.csv"
CLASS_METADATA = PROCESSED_DATA_DIR / "class_metadata.csv"
