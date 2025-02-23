import math 

# Exponential decay 
def scheduler_exponential_decay(
        epoch, 
        total_epochs, 
        initial_lr, 
        args = [-0.15] # Default to -0.15 for smaller #'s of epochs 
        ): 
    
    '''
    Purpose: learning rate scheduler for model training. 

    Arguments: 
    - epoch (int): epoch number. 
    - total_epochs (int): total number of epochs. 
    - initial_lr (float): initial learning rate for model. Specified as constant in main.py. 
    - args (list): list of arguents for learning rate scheduler. 

    Returns: 
    '''
    
    decay_rate = args[0]
    
    # Calculates step 
    step = epoch - 1
    
    new_lr = initial_lr * math.exp(decay_rate * step)
    print(f"Set learning rate to {round(new_lr, 8)}. Scheduler: exponential decay. \n")

    return new_lr