import torch, torchvision # type: ignore 
import torchvision.transforms.v2 as transforms # type: ignore 

def transform_original(image_size, dataset_path): 

    '''
    Purpose: Apply a transform and create a torchvision.datasets.ImageFolder object with transformed images. 

    Arguments: 
    - image_size (int): image size. 
    - dataset_path (str): path to the dataset. 

    Returns: 
    - torchvision.datasets.ImageFolder: ImageFolder object with transformed images. 
    '''

    image_size = int(image_size)

    # Transformation 
    transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), # Resize image
            transforms.PILToTensor(), # Changes input to Pytorch tensor format 
            transforms.ToDtype(torch.float32, scale = True), # Changes data type of tensor and normalizes pixel values 
        ])
    
    # Create dataset 
    out_dataset = torchvision.datasets.ImageFolder(
            dataset_path, 
            transform = transform,
        )
    
    return out_dataset 

def transform_dynamic(image_size, dataset_path): 

    '''
    Purpose: Apply a transform and create a torchvision.datasets.ImageFolder object with transformed images. 

    Arguments: 
    - image_size (int): image size. 
    - dataset_path (str): path to the dataset. 

    Returns: 
    - torchvision.datasets.ImageFolder: ImageFolder object with transformed images. 
    '''
    
    image_size = int(image_size) 

    # Transformation 
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=image_size, scale=(0.9, 1.0), ratio=(1.0, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=(-10, 10)),  
        transforms.PILToTensor(), # Changes input to Pytorch tensor format 
        transforms.ToDtype(torch.float32, scale = True), # Changes data type of tensor and normalizes pixel values 
    ])

    # Create dataset 
    out_dataset = torchvision.datasets.ImageFolder(
            dataset_path, 
            transform = transform,
        )
    
    return out_dataset 