import os
import shutil
import random
import math

def split_data(dataset_path, train_dir, val_dir, split_ratio):

    '''
    Purpose: Splits dataset into temporary train and validation datasets. 

    Arguments: 
    - dataset_path (str): path to the dataset. 
    - train_dir (str): name of the training dataset. 
    - val_dir (str): name of the validation dataset. 
    - split_ratio (float): proportion of images going to the training dataset. 

    Returns: None. 
    '''
    
    print("Splitting data into train and validation sets. ")

    # Seed 
    random.seed(37)
    
    # Error handling 
    if not os.path.exists(dataset_path):
        raise ValueError(f"Original dataset directory '{dataset_path}' does not exist.")
    
    # Create new directories 
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Iterates over all subdirectories in original dataset 
    for class_name in os.listdir(dataset_path):

        class_dir = os.path.join(dataset_path, class_name)
        
        # Check if not a directory 
        if not os.path.isdir(class_dir):
            print(f"Skipping '{class_dir}' as it is not a directory.")
            continue
        
        # Creates list of all images in subdirectory 
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        
        # Random shuffle 
        random.shuffle(images)
        
        # Calculate split 
        total_images = len(images)
        train_count = math.floor(split_ratio * total_images)
        
        # Create subsets of original (shuffled) directory 
        train_images = images[:train_count]
        val_images = images[train_count:]
        
        # Create class subdirectories 
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Copy training subset of images 
        for img in train_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy2(src, dst)
        
        # Copy validation subset of images 
        for img in val_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(val_class_dir, img)
            shutil.copy2(src, dst)
        
        print(f"Class '{class_name}': {train_count} images copied to training set, {len(val_images)} images copied to validation set")

    print("Dataset splitting completed successfully. ") 
    print()
    print()