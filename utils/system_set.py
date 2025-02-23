import platform
import torch, torchvision # type: ignore

def set_device(): 

    '''
    Purpose: Sets default device. 
    If on Mac, checks first for mps, then for cpu. 
    If on Windows, checks first for cuda, then for cpu. 

    Arguments: None. 

    Returns: 
    - torch.device: default device. 
    '''
        
    import platform
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