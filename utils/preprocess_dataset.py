import shutil 
from PIL import Image # type: ignore
from torchvision import transforms # type: ignore
import os 
import sys

# Checks if each image can be opened by Torch (test_load_image function). Deletes if not. Returns error count. 
def preprocess_images(dataset_name): 

    '''
    Purpose: Preprocess dataset to ensure that all images can be loaded in torch. 

    Arguments: 
    - dataset_name (str): name of the dataset. Path works too. 

    Returns: 
    - int: total number of errors found. 
    '''

    # Keeping count of errors found/files deleted
    error_count = 0

    # Maps original subdirectory name to desired name with short string
    name_dict = {
        'ferrari': 'ferrari', 
        'mclaren': 'mclaren', 
        'mercedes': 'mercedes',  
        'red': 'redBull', 
        'renault': 'renault', 
        'williams': 'williams'
    }

    # List of classes to include in final dataset 
    included_classes = [
        'ferrari', 
        'mclaren', 
        'mercedes', 
        'redBull', 
        'renault', 
        'williams'
    ]


    # Renames subdirectories 
    for root, dirs, _ in os.walk(dataset_name): 
        for dir in dirs: 
            for key, value in name_dict.items():
                if key in str(dir).lower(): 

                    old_path = os.path.join(root, dir)
                    new_path = os.path.join(root, value)

                    os.rename(old_path, new_path)
                    

    # Creates list of all subdirectories in dataset
    subdir_list = []
    for root, dirs, _ in os.walk(dataset_name): 
        for dir in dirs: 
            subdir_list.append(os.path.join(root, dir))

    print()
    print()
    # Delete datasets not in list of included classes (included_classes) 
    for subdir in subdir_list: 
        delete_dataset = True 
        for class_name in included_classes: 
            if subdir == os.path.join(dataset_name, class_name):
                delete_dataset = False 
                print("Kept directory", subdir)
        if delete_dataset == True: 
            shutil.rmtree(subdir)
            print("Deleted directory", subdir)
    print()
    print()

    # Creates NEW list of all subdirectories in dataset (we deleted some earlier) 
    subdir_list = []
    for root, dirs, _ in os.walk(dataset_name): 
        for dir in dirs: 
            subdir_list.append(os.path.join(root, dir))

    # Prints hello message 
    print_hello = True

    # Iterates through all files in dataset 
    for subdir in subdir_list: 
        index = 0 # Used for renaming individual files 
        file_paths = [os.path.join(subdir, fname) for fname in os.listdir(subdir)] # List of all file paths 

        total_count = 0
        for path in file_paths: 

            test_load = test_load_image(path)

            if test_load == True: 
                total_count += 1
                
            elif test_load == False: 
                # If there is an error, delete the file
                error_count += 1
                os.remove(path)
                total_count += 1
                continue

            # Hello message
            if print_hello == True: 
                print("-" * 49) 
                print("Removing images that are corrupted or unsupported")
                print("-" * 49) 
                print('\n\n')
                print_hello = False

            # Progress bar 
            percent = (total_count / (len(file_paths) - 1)) * 100
            bar = '#' * int(percent / 5) + '-' * (20 - int(percent / 5))

            # Use f-string alignment for consistent formatting
            print(f'\rProcessing Subdirectory: {subdir:<30} [{bar}] {percent:6.2f}%', end='')

            # Renaming files. Some of the old files had really messy names... 
            if sys.platform=="win32": 
                teamname = str(subdir).split('\\')[1]
            elif sys.platform == "darwin": 
                teamname = str(subdir).split('/')[1]

            suffix = str('.' + path.split('.')[-1])
            old_path = os.path.join(path)
            new_path = os.path.join(subdir, str(teamname + '_' + str(index) + suffix))
            index += 1
            os.rename(old_path, new_path)

        # Final print of progress bar 
        print(f'\rProcessing Subdirectory: {subdir:<30} [{"#" * 20}] 100.00%')

    # Returns total number of errors found 
    return error_count

# Tries to load image in Torch
def test_load_image(path):
        
    '''
    Purpose: Try to load the image as a tensor. 

    Arguments: 
    - path (str): path to the image. 

    Returns: 
    - bool: True if image was successfully loaded, False if not. 
    '''
        
    transform = transforms.ToTensor()
    img = None
    try:
        img = Image.open(path)
        img_tensor = transform(img)
        img.close()

        return True
    
    except:
        return False

# Checks if each file is in a supported format. Formats are listed in supported_formats. Returns error count. 
def check_files(dataset_name): 

    '''
    Purpose: Includes only files with supported formats (supported_formats)

    Arguments: 
    - dataset_name (str): name of the dataset 

    Returns: 
    - int: total number of files with invalid formats found. 
    '''

    # Some files were causing errors in TensorFlow. For simplicity, this function removes all files that are not formatted as a .jpg

    import os

    # Keeps count of total errors 
    error_count = 0

    supported_formats = (".jpg",) # List of supported formats

    # Checks the format of every file, deletes files that are not formatted as a .jpg 
    ct=0
    for root, _, files in os.walk(dataset_name):
        for file in files:

            # Removes hidden files
            if file.startswith('.'):
                path0 = os.path.join(root, file)
                #print(path0)
                os.remove(path0)
                continue

            ct+=1

            # Removes files that are not .jpg
            if not file.lower().endswith(supported_formats):
                path0 = os.path.join(root, file)
                #print(f"Unsupported file format found and ignored: {file} \r")
                error_count += 1
                os.remove(path0)

    return error_count 

# Counts number of files in each class 
def count_images(dataset_name): 

    '''
    Purpose: Counts images in all subdirectories. 

    Arguments: 
    - dataset_name (str): name of the dataset. 

    Returns: None. 
    '''
        
    import os

    # Maps original subdirectory name to desired name
    name_dict = {
        'alphatauri': 'alphatauri', 
        'aston': 'astonMartin', 
        'ferrari': 'ferrari', 
        'lotus': 'lotus', 
        'mclaren': 'mclaren', 
        'mercedes': 'mercedes', 
        'point': 'racingPoint', 
        'red': 'redBull', 
        'renault': 'renault', 
        'williams': 'williams'
    }               

    # Creates list of all subdirectories in dataset
    subdir_list = []
    for root, dirs, _ in os.walk(dataset_name): 
        for dir in dirs: 
            subdir_list.append(os.path.join(root, dir))
            
    # Iterates through all files in dataset 
    for subdir in subdir_list: 
        index = 0 # Used for renaming individual files 
        file_paths = [os.path.join(subdir, fname) for fname in os.listdir(subdir)] # List of all file paths 

        print(f"{subdir}", f"{len(file_paths)}", "images. ")
        