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
    for i in range(32):
        for j in range(32):
            print('Optimizing on: ',(i,j))
            energy2 = ndf(image, torch.Tensor([i,j]).view(2,-1))
            energy_diff = torch.norm(energy2-energy1)
            if energy_diff < min_diff:
                min_diff = energy_diff
                min_coord = torch.Tensor([i,j]).view(2,-1)
    return min_coord, min_diff

if __name__ == '__main__':
    transform_to_tensor = transforms.ToTensor()

    image1 = Image.open('./data/MNIST/overlapMNIST/train/07/122_07.png')
    image1 = transform_to_tensor(image1)

    image1_with_points, coordinates =  writePoints(image1.squeeze())

    image2 = Image.open('./data/MNIST/overlapMNIST/train/37/31_37.png')
    image2 = transform_to_tensor(image2)
    
    min_coords = []
    for coord in coordinates:
        coord = torch.Tensor(coord).view(2,-1)
        min_coord, min_diff = optimize(image1, coord, image2)
        min_coords.append(min_coord)
    
    image2 = image2.squeeze().numpy()
    for min_coord in min_coords:
        cv2.circle(image2, (int(min_coord[0,0].item()), int(min_coord[1,0].item())), radius = 1,  color=(0,255,0), thickness = 1)
    
    plt.imshow(image1_with_points, cmap='gray')

    plt.imshow(image2, cmap='gray')
    plt.show()





