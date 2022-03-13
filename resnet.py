#imports
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms, utils
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models
import constants
from dataloader_ndf import OverlapMNISTNDF
from constants import IMG_DIR
import torch.nn as nn

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 128)
model_ft.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias = False)
model_ft.maxpool = nn.Identity()
image_dataset = OverlapMNISTNDF(IMG_DIR, ToTensor(), 'train')
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size = 32, shuffle = False)
