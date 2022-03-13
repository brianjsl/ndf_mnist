from sqlalchemy import outparam
import torch
from dataloader_ndf import OverlapMNISTNDF
import torchvision.transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import time
from tqdm import tqdm
from constants import IMG_DIR
import copy
from neural_field import NeuralField
import matplotlib.pyplot as plt

model = torch.load('./checkpoints/chkpt_9.pt', map_location='cpu')
model.eval()

data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
])
image_dataset = OverlapMNISTNDF(IMG_DIR, data_transforms, 'train')
subset = torch.utils.data.Subset(image_dataset, [i for i in range(1024*0, 1024*1)])
dataloader = torch.utils.data.DataLoader(subset, batch_size = 32, shuffle = False, num_workers = 0)
iterable = iter(dataloader)
reconstructed = torch.zeros((32,32))
for i in range(32):
    ((image, coordinates), intensities) = next(iterable)
    output = model((image, coordinates))
    for j in range(32):
        print(coordinates[j])
        reconstructed[coordinates[j][1][0]][coordinates[j][0][0]] = output[j].item()
plt.imshow(reconstructed, cmap='gray')
plt.show()
