import matplotlib.pyplot as plt     

def graph(total_epochs, train_stats): 

    '''
    Purpose: Creates graphs with model performance. 

    Arguments: 
    - total_epochs (int): total number of epochs. 
    - train_stats (list): list of perfomance data from training. Contains train/validation accuracy/loss. 

    Returns: None. 
    '''
        
    x_points = []
    for i in range(total_epochs): 
        x_points.append(int(i + 1))

    plt.figure(figsize = (8, 6)) 
    plt.plot(x_points, train_stats[0], color='r', label='Train Loss')
    plt.plot(x_points, train_stats[2], color='b', label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend()
    plt.savefig('loss_graph.png')
    plt.show()


    plt.figure(figsize = (8, 6)) 
    plt.plot(x_points, train_stats[1], color='r', label='Train Accuracy')
    plt.plot(x_points, train_stats[3], color='b', label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Epoch")
    plt.legend()
    plt.savefig('accuracy_graph.png')
    plt.show()