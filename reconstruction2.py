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
import argparse
import PIL.Image as Image

def argparser():   
    '''
    Argparser Setup. 
    
    Arguments: 
        --image_num: number of image to run the reconstruction on in dataset.
    '''

    #initialize argparser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--image_class', type = str, 
                        default = '61',
                        help='class of image'
                        )
    parser.add_argument('--image_num', type = str, 
                        default = '0',
                        help='number of image'
                        )
    config = parser.parse_args()
    return config

if __name__ == '__main__':
    config = argparser()
    model = torch.load('./checkpoints/new/chkpt_39.pt', map_location='cpu')
    model.eval()

    data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5],[0.5])
    ])
    reconstructed = torch.zeros((32,32))
    image = Image.open('./data/MNIST/overlapMNIST/train/'+config.image_class+'/'+\
            config.image_num+'_'+config.image_class+'.png')
    image = data_transforms(image)
    image = torch.unsqueeze(image,0) 

    plt.figure(figsize=[8,4]);

    for i in tqdm(range(32)): 
        for j in range(32):
            coordinates = torch.tensor([i,j]).view(2,-1)
            output = model((image, coordinates))
            reconstructed[j][i] = output.item()
    plt.subplot(121); plt.imshow(image.squeeze(), cmap = 'gray'); plt.title('Original Image')
    plt.subplot(122); plt.imshow(reconstructed.squeeze(), cmap = 'gray'); plt.title('Reconstructed Image')
    plt.show()
