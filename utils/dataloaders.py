import torch # type: ignore
from torch.utils.data import random_split, DataLoader # type: ignore
import torchvision.transforms.v2 as transforms # type: ignore
from torchvision.datasets import ImageFolder # type: ignore
from torch.utils.data import ConcatDataset # type: ignore 

# To do: apply image transformations properly to subsets 

def create_dataloaders(
        dataset_names, 
        image_size, 
        batch_size, 
        train_transforms, 
        val_transforms
    ):

    '''
    Purpose: 
    - Applies list of transforms to train and validation datasets. 
    - Creates dataloader objects for both train and validation datasets. 

    Arguments: 
    - dataset_names (list): list of names of temporary train and validation directories. 
    - image_size (int): image size used for transforms. 
    - batch_size (int): batch size for training. 
    - train_transforms (list): list of transforms used for training data augmentation. 
    - val_transforms (list): list of transforms used for validation data. There should only be one function here. 

    Returns: 
    - list: list of two torch.utils.data.DataLoader objects. 
    '''

    train_path, val_path = dataset_names[0], dataset_names[1]
    
    train_dataloader = apply_transforms(
        batch_size, 
        train_path, 
        train_transforms,
        image_size
    )

    val_dataloader = apply_transforms(
        batch_size, 
        val_path, 
        val_transforms,
        image_size
    )

    return [train_dataloader, val_dataloader]


def apply_transforms(
        batch_size, 
        path_to_dataset, 
        transforms_list, # List of transformations. Original, and as many augmentations as required. 
        image_size
    ):  # type: ignore

    '''
    Purpose: 
    - Applies all transforms specified in transforms_list. 
    - Concatenates all images from each transform into a single dataset. 
    - Creates a torch.utils.data.DataLoader object with all images. 

    Arguments: 
    - batch_size (int): batch size. 
    - path_to_dataset (str): path to the dataset. 
    - transforms_list (list): list of transforms to apply. 
    - image_size (int): image size. 

    Returns: torch.utils.data.DataLoader: DataLoader object with all images. 
    '''

    # Creates random number generator on default device. Reproducible with seed 37 
    test_tensor = torch.tensor([1, 2, 3])
    default_device = test_tensor.device
    generator = torch.Generator(device = default_device).manual_seed(37)

    imageFolder_list = []
    for function in transforms_list: 
        out = function(image_size, path_to_dataset)
        imageFolder_list.append(out)
        
    master_dataset = ConcatDataset(imageFolder_list)

    # Creates two DataLoader objects for train and validation data 
    dataloader = torch.utils.data.DataLoader(master_dataset, batch_size = batch_size, shuffle = True, generator = generator)
    
    # Returns the DataLoader object 
    return dataloader 