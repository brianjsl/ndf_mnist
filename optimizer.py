import torch
from dataloader_ndf import OverlapMNISTNDF
import torchvision.transforms as transforms
import torchvision.transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from neural_field import NeuralField
from constants import IMG_DIR
import matplotlib.pyplot as plt
import PIL.Image as Image
from writePoints import writePoints
import cv2
import argparse
from tqdm import tqdm

data_transforms = transforms.Compose([
                transforms.Normalize([0.5],[0.5])
    ])

def ndf(image, coordinate):
    '''
    Params:
    @image: (1,32,32) Tensor Image
    @coordinate: (2,1) Coordinate 
    '''
    activations = {}

    def getActivation(name): 
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook 

    image = data_transforms(image)
    image = torch.unsqueeze(image, 0)
    coordinate = torch.unsqueeze(coordinate,0)

    model = torch.load('./checkpoints/chkpt_29.pt', map_location='cpu')

    h1 = model.linear_relu_stack[1].register_forward_hook(getActivation('layer1'))
    h2 = model.linear_relu_stack[3].register_forward_hook(getActivation('layer2'))
    h3 = model.linear_relu_stack[5].register_forward_hook(getActivation('layer3'))

    output = model((image, coordinate))
    energy = torch.cat((activations['layer1'], activations['layer2'], activations['layer3']), 1)

    h1.remove()
    h2.remove()
    h3.remove()
    return energy

def optimize(target_image, target_coord, image):
    '''
    Given an input image and coordinate finds the energy minimized coordinate in image2.
    Params1: 
    @target_image: image you want to minimize energy to
    @target_coord: corresponding coordinate
    @image: image you sample over
    '''
    energy1 = ndf(target_image, target_coord)
    min_diff = float('inf')
    min_coord = None
    for i in tqdm(range(32)):
        for j in range(32): 
            energy2 = ndf(image, torch.Tensor([i,j]).view(2,-1))
            energy_diff = torch.norm(energy2-energy1)
            if energy_diff < min_diff:
                min_diff = energy_diff
                min_coord = torch.Tensor([i,j]).view(2,-1)
    return min_coord, min_diff

def argparser():
    '''
    Initializes argparser. 

    Arguments: 
    --image1_class: class of image 1
    --image1_num: number of image 1
    --image2_class: class of image 2
    --image2_num: class of image2
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--image1_class', type = str, 
                        default = 00,
                        help='class of image 1'
                        )
    parser.add_argument('--image1_num', type = str, 
                        default = 0,
                        help='num of image 1'
                        )
    parser.add_argument('--image2_class', type = str, 
                        default = 00,
                        help='class of image 2'
                        )
    parser.add_argument('--image2_num', type = str, 
                        default = 0,
                        help='num of image 2'
                        )
    config = parser.parse_args()
    return config

if __name__ == '__main__':
    config = argparser()
    transform_to_tensor = transforms.ToTensor()

    image1 = Image.open('./data/MNIST/overlapMNIST/train/'+config.image1_class+'/'+config.image1_num+'_'\
        +config.image1_class+'.png')
    image1 = transform_to_tensor(image1)

    image1_with_points, coordinates =  writePoints(image1.squeeze())


    image2 = Image.open('./data/MNIST/overlapMNIST/train/'+config.image2_class+'/'+config.image2_num+'_'\
        +config.image2_class+'.png')
    image2 = transform_to_tensor(image2)
    
    min_coords = []
    for coord in coordinates:
        coord = torch.Tensor(coord).view(2,-1)
        min_coord, min_diff = optimize(image1, coord, image2)
        min_coords.append(min_coord)
    
    image2 = image2.squeeze().numpy()
    for min_coord in min_coords:
        cv2.circle(image2, (int(min_coord[0,0].item()), int(min_coord[1,0].item())), radius = 1,  color=(0,255,0), thickness = 1)
    
    plt.figure(figsize=[8,4]);
    plt.subplot(121); plt.imshow(image1_with_points.squeeze(), cmap = 'gray'); plt.title('Image 1 with labeled points')
    plt.subplot(122); plt.imshow(image2.squeeze(), cmap = 'gray'); plt.title('Image 2 with corresponding points')
    plt.show()
    





