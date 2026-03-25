import os
import sys

# Add the 'ui' directory to the Python path so local imports work correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# HuggingFace Spaces expects an 'app.py' at the root of the repository.
# This file bridges the root directory with the actual ui/app.py Streamlit script.
with open("ui/app.py") as f:
    exec(f.read())
