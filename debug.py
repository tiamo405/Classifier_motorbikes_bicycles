import cv2
import os
import torch
import numpy as np 
from torch.utils import data
from torchvision import transforms

transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.CenterCrop(0.5),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
path_data = 'data/train'
for label in os.listdir(path_data) :
    for fname in os.listdir(os.path.join(path_data, label)) :
        path_image = os.path.join(path_data, label, fname)
        image = cv2.imread(path_image)
        try:
            image = transform(image)    
        except :
            print(path_image)
            # os.remove(path_image)