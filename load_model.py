import torch 
from utils import *
import torchvision.transforms.v2 as transforms # type: ignore 
import os 
from PIL import Image
import argparse

# Image size 
IMAGE_SIZE = 256

# Class labels (alphabetical)
CLASS_LABELS = ["ferrari", "mclaren", "mercedes", "redBull", "renault", "williams"]


# Set up command line argument parser 
ap = argparse.ArgumentParser()

# Adding arguments 
ap.add_argument("-d", "--test_set", required = True, help = "Path to test set directory")

# Store command line arguments in variables 
args = vars(ap.parse_args())
image_dir = str(args['model'])

# Load model, set to evaluation mode 
model = torch.load("model.pth")
model.eval() 


# Setting default device 
default_device = set_device()


# Resizing transformation 
transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # Resize image
            transforms.PILToTensor(), # Changes input to Pytorch tensor format 
            transforms.ToDtype(torch.float32, scale = True), # Changes data type of tensor and normalizes pixel values 
        ])


for img_name in os.listdir(image_dir): 
    img_path = os.path.join(image_dir, img_name)

    if not img_name.lower().endswith(('jpg')): 
        continue

    image = Image.open(img_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0) # Adds batch dimension 

    with torch.no_grad(): 
        output = model(image)
        probs = torch.nn.functional.softmax(output[0], dim = 0) 
        predicted_class = torch.argmax(probs).item()

    print(f"Image: {img_name} | Predicted Class: {CLASS_LABELS[predicted_class]} | Confidence: {probs[predicted_class]:.4f}")