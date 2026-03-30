import json
import os

# 1. Setup the data
_ROOT = os.path.abspath(os.path.dirname(__file__))
_DEFS_PATH = os.path.join(_ROOT, 'mindice_defs.json')

with open(_DEFS_PATH, 'r') as f:
    DEFAULT_DEFINITIONS = json.load(f)

# 2. Import the function from core
from .core import mindice

# 3. Explicitly define what is "public"
__all__ = ['mindice', 'DEFAULT_DEFINITIONS']
