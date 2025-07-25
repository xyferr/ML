"""
Configuration settings for ML projects.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Model settings
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CV_FOLDS = 5

# Data processing settings
DEFAULT_SCALING_METHOD = "standard"  # Options: "standard", "minmax", "robust"
DEFAULT_ENCODING_METHOD = "onehot"  # Options: "onehot", "label", "target"

# Visualization settings
FIGURE_SIZE = (10, 6)
PLOT_DPI = 300
PLOT_STYLE = "seaborn-v0_8"

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Model performance thresholds
MIN_ACCURACY_THRESHOLD = 0.8
MIN_PRECISION_THRESHOLD = 0.75
MIN_RECALL_THRESHOLD = 0.75
MIN_F1_THRESHOLD = 0.8

# Feature engineering settings
MAX_FEATURES_FOR_CORRELATION = 50
CORRELATION_THRESHOLD = 0.95
VARIANCE_THRESHOLD = 0.01

# Hyperparameter tuning settings
N_ITER_RANDOM_SEARCH = 100
N_JOBS_PARALLEL = -1
SCORING_METRIC = "accuracy"  # For classification

# MLflow settings (if using)
MLFLOW_EXPERIMENT_NAME = "ML_Experiments"
MLFLOW_TRACKING_URI = "file:./mlruns"

# Common file extensions
SUPPORTED_DATA_FORMATS = [".csv", ".xlsx", ".json", ".parquet", ".pkl"]
MODEL_FORMATS = [".pkl", ".joblib", ".h5", ".pt", ".pth"]

# Environment-specific settings
class DevelopmentConfig:
    DEBUG = True
    TESTING = False
    DATABASE_URI = "sqlite:///dev.db"

class ProductionConfig:
    DEBUG = False
    TESTING = False
    DATABASE_URI = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/prod")

class TestingConfig:
    DEBUG = False
    TESTING = True
    DATABASE_URI = "sqlite:///test.db"

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
