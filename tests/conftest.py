import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')

for path in [PROJECT_ROOT, SRC_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)
