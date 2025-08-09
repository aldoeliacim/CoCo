import os
from pathlib import Path

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "out"
RESULTS_DIR = PROJECT_ROOT / "results"  # Main results directory
LOGS_DIR = OUT_DIR / "logs"  # Temporary processing logs

# Model storage paths
MODELS_DIR = DATA_DIR / "models"
VISIONTHINK_MODEL_PATH = MODELS_DIR / "VisionThink-General"
YOLO_PANEL_MODEL_PATH = MODELS_DIR / "best.pt"

# Data paths
DEFAULT_INPUT_DIR = DATA_DIR / "fantomas" / "raw"

# Output directories (everything goes to out/ first, then moved to results/)
DEFAULT_OUTPUT_DIR = OUT_DIR / "analysis"
DEFAULT_FINETUNING_DIR = OUT_DIR / "finetuning"

# Legacy support (for backward compatibility)
ARCHIVE_DIR = PROJECT_ROOT / "archive"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# VisionThink Model Configuration
VISIONTHINK_CONFIG = {
    "max_new_tokens": 800,              # Increased for detailed character analysis
    "temperature": 0.3,                 # Sampling temperature (lower = more focused)
    "do_sample": True,                  # Enable sampling
    "torch_dtype": "float16",           # Model precision

    # Enhanced character analysis settings
    "character_analysis_enabled": True,  # Enable enhanced character tracking
    "adaptive_resolution": True,         # Use VisionThink adaptive resolution
    "multi_turn_analysis": True,         # Enable multi-turn reasoning
    "character_verification": True,      # Enable LLM-as-Judge character verification
}

# Character Registry Configuration
CHARACTER_REGISTRY_CONFIG = {
    "max_context_characters": 5,        # Maximum characters to include in context
    "significance_threshold": 0.3,      # Minimum significance for character classification
    "confidence_threshold": 0.5,        # Minimum confidence for character registration
    "relationship_threshold": 2,        # Minimum co-appearances for relationship
    "visual_consistency_check": True,   # Enable visual consistency validation
}

# YOLO Panel Detection Configuration
YOLO_CONFIG = {
    "confidence_threshold": 0.15,      # Lowered confidence threshold for better detection
    "iou_threshold": 0.4,              # Slightly lowered IoU threshold for overlapping panels
}

# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

PROCESSING_CONFIG = {
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "auto",
    "max_panels_per_page": 50,          # Maximum panels to process per page
    "panel_min_area": 500,              # Lowered minimum area for valid panels (pixels²)

    # Enhanced character tracking settings
    "enhanced_character_tracking": True, # Enable SOTA character tracking
    "character_context_size": 15,        # Number of characters to maintain in context
    "adaptive_resolution_threshold": 0.6, # Confidence threshold for high-res request

    "output_subdirs": {                 # Subdirectories for organized output
        "xml": "xml",
        "annotated_panels": "annotated_panels",
        "character_data": "character_data",    # Character registry and reports
        "visualizations": "visualizations"     # Character analysis visualizations
    }
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Global logging level - can be overridden at runtime
_GLOBAL_LOG_LEVEL = "INFO"  # Default: INFO level

def set_log_level(level: str):
    """Set the global logging level for all modules."""
    global _GLOBAL_LOG_LEVEL
    _GLOBAL_LOG_LEVEL = level.upper()

    # Update all existing CoCo loggers
    import logging
    for name in logging.Logger.manager.loggerDict:
        if name.startswith(('main', 'analysis', 'visionthink', 'xml_validator')):
            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, _GLOBAL_LOG_LEVEL, logging.INFO))

def get_log_level() -> str:
    """Get the current global logging level."""
    return _GLOBAL_LOG_LEVEL

LOGGING_CONFIG = {
    "level": get_log_level,              # Use dynamic level function
    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "max_file_size_mb": 10,             # Maximum log file size in MB
    "backup_count": 5,                  # Number of backup log files to keep
    "console_output": True,             # Enable console logging
}

# =============================================================================
# ARCHIVE CONFIGURATION
# =============================================================================

ARCHIVE_CONFIG = {
    "compression_enabled": True,        # Use compression by default
}

# =============================================================================
# DIRECTORY INITIALIZATION
# =============================================================================

def initialize_directories():
    """Initialize all required directories."""
    directories = [
        DATA_DIR, OUT_DIR, LOGS_DIR, MODELS_DIR,
        DEFAULT_OUTPUT_DIR, RESULTS_DIR, DEFAULT_INPUT_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_log_file_path(log_name: str = "main") -> Path:
    """Get the path for a log file."""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d")
    return LOGS_DIR / f"{log_name}_{timestamp}.log"


def get_timestamped_results_dir() -> Path:
    """Get timestamped results directory for today's results."""
    from datetime import datetime

    today = datetime.now().strftime("%Y-%m-%d")
    return RESULTS_DIR / today


def get_module_results_dir(module_name: str) -> Path:
    """Get timestamped module-specific results directory."""
    timestamped_dir = get_timestamped_results_dir()
    return timestamped_dir / module_name


def get_base_settings() -> dict:
    """Get base configuration settings as a dictionary."""
    return {
        'project_root': PROJECT_ROOT,
        'data_dir': DATA_DIR,
        'out_dir': OUT_DIR,
        'results_dir': RESULTS_DIR,
        'logs_dir': LOGS_DIR,
        'models_dir': MODELS_DIR,
        'default_output_dir': DEFAULT_OUTPUT_DIR,
        'default_input_dir': DEFAULT_INPUT_DIR,
        'visionthink_model_path': VISIONTHINK_MODEL_PATH,
        'yolo_panel_model_path': YOLO_PANEL_MODEL_PATH
    }