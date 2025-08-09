"""
Standardized project imports for CoCo.
This module provides consistent path setup for all project modules.
"""

import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

def setup_project_paths():
    """Add project paths to sys.path for consistent imports."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(PROJECT_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Auto-setup when imported
setup_project_paths()

# Common imports that all modules need
from config.settings import *
from config.logger import setup_logger
from config.settings import set_log_level
