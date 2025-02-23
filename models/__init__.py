"""
models package
==============

This package contains modules for creating and training various image classification models.

Available Modules:
- model_1: Handles the creation and training of Model 1.
- model_2: Handles the creation and training of Model 2.
- model_3: Handles the creation and training of Model 3.

Usage:
    from models import model_1, model_2, model_3
"""

# Import the key functions/classes from each module for convenient access 
from .model_resnet50 import create_model_resnet
from .model_Cv1 import create_model_Cv1
from .model_Cv2 import create_model_Cv2
from .model_Cv3 import create_model_Cv3

# Define a list of all publicly available items to restrict what gets imported
__all__ = [
    "create_model_resnet",
    "create_model_Cv1", 
    "create_model_Cv2", 
    "create_model_Cv3", 
]
