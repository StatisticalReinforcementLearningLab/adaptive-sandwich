"""
This file is used to add the Oralytics src directory to the sys.path so that the tests can import
modules from there and more importantly the imports in those files need not be modified.
"""

import sys
from pathlib import Path

# Adjust the path to point to your src directory
src_path = Path(__file__).resolve().parent / "oralytics_sample_data" / "Archive" / "src"
sys.path.insert(0, str(src_path))
