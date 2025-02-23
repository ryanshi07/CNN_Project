import torch.nn as nn # type: ignore 

def create_model_Cv3(total_classes): 
    model = nn.Sequential(

        # Conv 1: (256, 256, 3) -> (126, 126, 32)
        nn.ZeroPad2d((0, 1, 0, 1)),
        nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 7, stride = 2), 
        nn.BatchNorm2d(32),
        nn.ReLU(inplace = True), 

        # Conv 2: (126, 126, 32) -> (63, 63, 64) 
        nn.ZeroPad2d((0, 1, 0, 1)),
        nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2), 
        nn.BatchNorm2d(64),
        nn.ReLU(inplace = True), 

        # Conv 3: (63, 63, 64) -> (30, 30, 128) 
        nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2), 
        nn.BatchNorm2d(128),
        nn.ReLU(inplace = True), 
        nn.AvgPool2d(kernel_size = 2, stride = 1), 

        # Conv 4: (30, 30, 128) -> (15, 15, 256)
        nn.ZeroPad2d((0, 1, 0, 1)), 
        nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2), 
        nn.BatchNorm2d(256),
        nn.ReLU(inplace = True), 

        # Flatten 
        nn.AdaptiveAvgPool2d((1, 1)), 
        nn.Flatten(), 

        # FC 1: (256) -> (128) 
        nn.Dropout(p = 0.25),
        nn.Linear(in_features = 256, out_features = 128),
        nn.ReLU(inplace = True),

        # FC 2: (128) -> (total_classes) 
        nn.Dropout(p = 0.25),
        nn.Linear(in_features = 128, out_features = total_classes),
    )
    
    return model 