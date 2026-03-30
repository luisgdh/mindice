import json
import os

# --- 1. Metadata ---
__version__ = "1.45"
__author__ = "Luis Gabriel Dahmer Hahn"

# 2. Setup the data
_ROOT = os.path.abspath(os.path.dirname(__file__))
_DEFS_PATH = os.path.join(_ROOT, 'mindice_defs.json')

with open(_DEFS_PATH, 'r') as f:
    DEFAULT_DEFINITIONS = json.load(f)

# 3. Import the function from core
from .core import mindice
