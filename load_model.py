import torch 
from utils import *
import torchvision.transforms.v2 as transforms # type: ignore 
import os 
import numpy as np
from PIL import Image

# Test set directory 
IMAGE_DIR = "C:\\Users\\ryany\\Desktop\\custom_cnn-main\\test_set"

# Image size 
IMAGE_SIZE = 256

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

CLASS_LABELS = ["ferrari", "mclaren", "mercedes", "redBull", "renault", "williams"]

for img_name in os.listdir(IMAGE_DIR): 
    img_path = os.path.join(IMAGE_DIR, img_name)

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