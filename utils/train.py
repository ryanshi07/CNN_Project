import torch, torchvision # type: ignore 
import torchvision.transforms.v2 as v2 # type: ignore 
from torch.utils.data import DataLoader # type: ignore
import torch.nn as nn # type: ignore 
import torch.nn.functional # type: ignore 

def train_model(
        model, 
        dataloaders, 
        loss_function, 
        optimizer, 
        total_epochs, 
        initial_lr, 
        learning_rate_scheduler, 
        scheduler_args
        ):
    
    '''
    Purpose: Trains model. 

    Arguments: 
    - model (torch.nn.Sequential): the model. 
    - dataloaders (list): list of two torch.utils.data.DataLoader objects, one for training, one for validation. 
    - loss_function (function): loss function. 
    - optimizer (function): optimizer. 
    - total_epochs (int): total number of epochs. 
    - initial_lr (float): initial learning rate. 
    - learning_rate_scheduler (function): learning rate scheduler function. 
    - scheduler_args (list): list of arguments for learning rate scheduler. 

    Returns: 
    - list: [epoch_train_losses, epoch_train_accuracies, epoch_val_losses, epoch_val_accuracies]. This is used as the performance data to create graphs. 
    '''

    train, validation = dataloaders[0], dataloaders[1]

    # Print number of batches 
    print(f"Train batches = {len(train)}")
    print(f"Validation batches = {len(validation)}")

    print(f"Initial learning rate: {initial_lr:.8f}")

    print()
    print()

    # Set model to training mode 
    model.train()

    epoch_val_losses = []
    epoch_val_accuracies = []
    epoch_train_losses = []
    epoch_train_accuracies = []

    for i in range(total_epochs):

        batch_losses = []
        batch_accuracies = []

        print(f"=== Epoch {i+1} ===")

        model.train() # Set model to train mode 

        # Training phase epoch 
        # Fetches one batch from train at a time 
        for (image_batch, label_batch) in train:

            predictions = model(image_batch) # Raw outputs of model. [batch_size, num_classes]

            loss = loss_function(predictions, label_batch) # Computed loss for the batch 
            loss.backward() # Performs back propagation on loss

            optimizer.step() # Updates model parameters based on calculated gradients 
            optimizer.zero_grad() # Clears gradients 

            batch_losses.append(float(loss)) # Stores loss value 
            batch_accuracies.append(float(sum(predictions.argmax(1) == label_batch))/len(label_batch)) # Calculates and stores accuracy value 

            cur_loss = sum(batch_losses)/len(batch_losses) # Calculates running average of batch losses 
            cur_acc = sum(batch_accuracies)/len(batch_accuracies) # Calculates running average of batch accuracies 

            # Print progress 
            print("Train:", end="\t\t")
            print(f"Batch: {len(batch_losses)}", end="\t")
            print(f"Loss: {round(cur_loss, 4)}", end="\t")
            print(f"Accuracy: {round(cur_acc, 4)}", end="\r")

        print()

        epoch_accuracy = sum(batch_accuracies) / len(batch_accuracies)
        epoch_train_accuracies.append(epoch_accuracy)

        epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_train_losses.append(epoch_loss)


        batch_losses = []
        batch_accuracies = []

        model.eval() # Set model to evaluation mode 

        # Validation phase epoch 
        # Fetches one batch from validation at a time 
        for (image_batch, label_batch) in validation:

            with torch.no_grad():

                predictions = model(image_batch) # Raw outputs of model. [batch_size, num_classes]

                # Softmax: logits --> probabilities 
                probabilities = torch.nn.functional.softmax(predictions, dim = 1)

                loss = loss_function(predictions, label_batch) # Computed loss for the batch 

                batch_losses.append(float(loss)) # Stores loss value 
                predicted_classes = probabilities.argmax(1)  # Use probabilities for predictions
                batch_accuracies.append(float(sum(predicted_classes == label_batch)) / len(label_batch))


                cur_loss = sum(batch_losses)/len(batch_losses) # Calculates running average of batch losses 
                cur_acc = sum(batch_accuracies)/len(batch_accuracies) # Calculates running average of batch accuracies 

                # Print progress 
                print("Validation:", end="\t\t")
                print(f"Batch: {len(batch_losses)}", end="\t")
                print(f"Loss: {round(cur_loss, 4)}", end="\t")
                print(f"Accuracy: {round(cur_acc, 4)}", end="\r")

        print()

        epoch_accuracy = sum(batch_accuracies) / len(batch_accuracies)
        epoch_val_accuracies.append(epoch_accuracy)

        epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_val_losses.append(epoch_loss)

        # Set new learning rate according to scheduler 
        epoch = i + 1

        # Calculates and sets new learning rate according to scheduler 
        new_lr = learning_rate_scheduler(epoch, total_epochs, initial_lr, args = scheduler_args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    return [epoch_train_losses, epoch_train_accuracies, epoch_val_losses, epoch_val_accuracies]