import shutil
import os
import torch # type: ignore
import platform

def set_device(): 

    '''
    Purpose: Sets default device. 
    If on Mac, checks first for mps, then for cpu. 
    If on Windows, checks first for cuda, then for cpu. 

    Arguments: None. 

    Returns: 
    - torch.device: default device. 
    '''

    user_os = platform.system()

    if user_os == "Darwin":  # macOS

        if torch.backends.mps.is_available():
            device = torch.device("mps")  # Metal Performance Shaders (MPS)
            torch.set_default_device(torch.device("mps"))
            print("Using MPS (Metal Performance Shaders) as the device.")
        else:
            device = torch.device("cpu")
            torch.set_default_device(torch.device("cpu"))
            print("MPS is not available. Using CPU as the device.")
            
    elif user_os == "Windows" or user_os == "Linux":  # Windows or Linux

        if torch.cuda.is_available():
            device = torch.device("cuda")  # CUDA
            torch.set_default_device(torch.device("cuda"))
            print(f"Using CUDA as the device. CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            torch.set_default_device(torch.device("cpu"))
            print("CUDA is not available. Using CPU as the device.")

    else:
        device = torch.device("cpu")
        torch.set_default_device(torch.device("cpu"))
        print("Unsupported OS. Using CPU as the device.") 

    return device 

def delete_datasets(base_path = '.', delete_dirs = []):
    
    '''
    Purpose: Deletes temporary training and validation datasets. Original dataset is not altered. 

    Arguments: 
    - base_path (str): the base path of the program. Defaults to '.'. 
    - delete_dirs (list): list of directory names to be deleted. 

    Returns: None. 
    '''
    
    # Attempt deletion of each directory 
    for dir_name in delete_dirs:

        dir_path = os.path.join(base_path, dir_name)

        if os.path.exists(dir_path):

            if os.path.isdir(dir_path):

                try:
                    shutil.rmtree(dir_path)
                    print(f"Successfully deleted directory: {dir_path}")
                except Exception as e:
                    print(f"Error deleting directory '{dir_path}': {e}")

            else:
                print(f"Path exists but is not a directory: {dir_path}")

        else:
            print(f"Directory does not exist: {dir_path}")

# Checks terminal size. Narrow terminal may cause formatting issues. 
def check_terminal_size(): 

    '''
    Purpose: Checks that the user's terminal size is large enough to display text properly. 

    Arguments: None. 

    Returns: None. 
    '''

    # If terminal size is too small, some messages might appear wrong 

    import shutil
    required_width = 100
    term_width = shutil.get_terminal_size().columns
    if term_width < required_width:
        print("Terminal size too small, please resize (minimum width = 100)")
        exit()