import collections
import torch.nn as nn # type: ignore
from torchvision.models import resnet50, ResNet50_Weights # type: ignore

def create_model_resnet(total_classes):

    # Transfer ResNet50 weights 
    weights = ResNet50_Weights.DEFAULT
    res = resnet50(weights=weights)

    # Don't train ResNet50 weights 
    for param in res.parameters():
        param.requires_grad = False

    # Replace final FC with identity 
    res.fc = nn.Identity()

    # New final layer in Sequential 
    model = nn.Sequential(collections.OrderedDict([
        ('resnet', res), # Output: 2048 
        ('final', nn.Linear(in_features = 2048, out_features = total_classes)),
    ]))

    return model
