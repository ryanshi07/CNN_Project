# Importing other libraries 
import argparse 
import torch # type: ignore

# Importing my functions 
from utils import * 
from models import *

BATCH_SIZE = 32 # Batch size for all training 
CLASSES = 6 # Number of classes 
INITIAL_LR = 0.001 # Initial learning rate 
TRAIN_SPLIT = 0.7 # Proportion of data going into train set 

# Create a list with all transforms to be used for data augmentation 
train_transforms = [transform_dynamic, transform_original]
val_transforms = [transform_original]

def main(
    create_model, 
    total_epochs, 
    classes, 
    image_size, 
    batch_size, 
    train_transforms, 
    val_transforms, 
    loss_function, 
    optimizer_function, 
    initial_lr, 
    lr_function, 
    lr_function_params, 
    ): 

    # Create model 
    model = create_model(classes)

    # Create dataloaders 
    dataloaders = create_dataloaders(
        dataset_names = [train_dir, val_dir], 
        image_size = image_size, 
        batch_size = batch_size, 
        train_transforms = train_transforms, 
        val_transforms = val_transforms
    )

    # Set up loss function  
    loss_function = loss_function()
    
    # Set up optimizer 
    optimizer = optimizer_function(model.parameters(), lr = initial_lr)

    # train_stats: [train_losses, train_accuracies, val_losses, val_accuracies] 
    train_stats = train_model(
        model, 
        dataloaders, 
        loss_function, 
        optimizer, 
        total_epochs, 
        initial_lr, 
        lr_function, 
        lr_function_params
    )
    
    # Graph results 
    graph(total_epochs, train_stats)

    # Print model architecture 
    print("Model Architecture: ")
    print(model)

    print()

    # Delete temporary train and val datasets 
    delete_datasets(base_path = '', delete_dirs = ['dataset_train', 'dataset_val'])

    return model 



if __name__ == "__main__": 

    print()
    print()

    # Set default device based on os and device availability 
    default_device = set_device()

    # Check terminal size 
    check_terminal_size() 

    # Checking that default device is properly set 
    test_tensor = torch.tensor([1, 2, 3]) 
    print(f"Device used: {test_tensor.device}") 

    # Set random seed so that results are reproducible.
    torch.manual_seed(37)

    print()
    print()

    # Set up command line argument parser 
    ap = argparse.ArgumentParser()

    # Adding arguments 
    ap.add_argument("-d", "--dataset", required = True, help = "Path to dataset directory")
    ap.add_argument("-m", "--model", required = True, help = "Enter model name. See README.md for model names. ") 
    ap.add_argument("-e", "--epochs", required = True, help = "Number of epochs")
    ap.add_argument("-p", "--preprocess", required = False, help = "Used only in testing, don't use this argument")

    # Store command line arguments in variables 
    args = vars(ap.parse_args())
    model_name = str(args['model'])
    total_epochs = int(args['epochs'])
    dataset_name = str(args['dataset'])

    # Extract preprocess argument 
    if not args['preprocess'] == None: 
        try: 
            preprocess = eval(args['preprocess'])
        except: 
            print("Preprocess argument is invalid, try again")
            exit()
    else: 
        preprocess = False


    # Preprocess dataset: removes corrupted or otherwise unsupported files 
    if preprocess == True: 

        print("Preprocessing dataset") 
        print("Initial count: ")
        count_images(dataset_name)

        error_count = preprocess_images(dataset_name) 
        print("Found and removed", error_count, "images that could not be converted to a tensor. ")

        bad_format_ct = check_files(dataset_name)
        print("Found and removed", bad_format_ct, "files with unsupported formats. ")

        print("\n\nFinal count: ")
        count_images(dataset_name)

    elif preprocess == False: 
        print("Skipped preprocessing")

    print()
    print()


    # Set up dataset split 
    train_dir = 'dataset_train'
    val_dir = 'dataset_val'
    
    # Split the dataset
    split_data(dataset_name, train_dir, val_dir, TRAIN_SPLIT)


    # Create and train models     
    if model_name == 'res': 
        model = main(
            create_model = create_model_resnet, 
            total_epochs = total_epochs, 
            classes = CLASSES, 
            image_size = 224, 
            batch_size = BATCH_SIZE, 
            train_transforms = train_transforms, 
            val_transforms = val_transforms, 
            loss_function = torch.nn.CrossEntropyLoss, 
            optimizer_function = torch.optim.Adam, 
            initial_lr = INITIAL_LR, 
            lr_function = scheduler_exponential_decay, 
            lr_function_params = [-0.20]
            ) 
        
        torch.save(model, "model_res.pth")
        
    elif model_name == 'Cv1': 
        model = main(
            create_model = create_model_Cv1, 
            total_epochs = total_epochs, 
            classes = CLASSES, 
            image_size = 256, 
            batch_size = BATCH_SIZE, 
            train_transforms = train_transforms, 
            val_transforms = val_transforms, 
            loss_function = torch.nn.CrossEntropyLoss, 
            optimizer_function = torch.optim.Adam, 
            initial_lr = INITIAL_LR, 
            lr_function = scheduler_exponential_decay, 
            lr_function_params = [-0.20]
            ) 
        
        torch.save(model, "model_Cv1.pth")

    elif model_name == 'Cv2': 
        model = main(
            create_model = create_model_Cv2, 
            total_epochs = total_epochs, 
            classes = CLASSES, 
            image_size = 256, 
            batch_size = BATCH_SIZE, 
            train_transforms = train_transforms, 
            val_transforms = val_transforms, 
            loss_function = torch.nn.CrossEntropyLoss, 
            optimizer_function = torch.optim.Adam, 
            initial_lr = INITIAL_LR, 
            lr_function = scheduler_exponential_decay, 
            lr_function_params = [-0.20]
            ) 
        
        torch.save(model, "model_Cv2.pth")
        
    elif model_name == 'Cv3': 
        model = main(
            create_model = create_model_Cv3, 
            total_epochs = total_epochs, 
            classes = CLASSES, 
            image_size = 256, 
            batch_size = BATCH_SIZE, 
            train_transforms = train_transforms, 
            val_transforms = val_transforms, 
            loss_function = torch.nn.CrossEntropyLoss, 
            optimizer_function = torch.optim.Adam, 
            initial_lr = INITIAL_LR, 
            lr_function = scheduler_exponential_decay, 
            lr_function_params = [-0.20]
            ) 
        
        torch.save(model, "model_Cv3.pth")
        
    else: 
        print("Invalid model name. Please try again. ")
        delete_datasets(base_path = '', delete_dirs = ['dataset_train', 'dataset_val'])
        exit()


