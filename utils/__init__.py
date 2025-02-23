"""
utils package
=============

This package contains utility functions for preprocessing data and managing training schedules.

Available Modules:
- preprocess: Contains functions for data preprocessing.
- schedulers: Contains functions for scheduling and optimization.
"""

# Functions/classes to import 
from .preprocess_dataset import preprocess_images, check_files, count_images
from .learning_rate_schedulers import scheduler_exponential_decay
from .train import train_model
from .transforms import transform_original, transform_dynamic
from .system import set_device, delete_datasets, check_terminal_size
from .plot import graph
from .split_dataset import split_data
from .dataloaders import create_dataloaders

# What gets importedw with from utils import * 
__all__ = [

    # .preprocess_dataset 
    "preprocess_images", 
    "check_files", 
    "count_images", 

    # .learning_rate_schedulers 
    "scheduler_exponential_decay", 

    # .train 
    "train_model", 

    # .transforms 
    "transform_original", 
    "transform_dynamic", 

    # .system
    "set_device", 
    "delete_datasets", 
    "check_terminal_size", 

    # .plot 
    "graph", 

    # .dataloaders
    "create_dataloaders", 

    # .split_dataset
    "split_data", 

]


