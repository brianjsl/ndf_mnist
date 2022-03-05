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

import constants
from constants import IMG_DIR

class OverlapMNIST(Dataset):
    '''
    Dataset of Overlapping MNIST Images. 
    Images returned as (h,w,3) size images.
    '''
    def __init__(self,
                 directory: str,
                 transforms: None,
                 set_name: str,
                ):
        '''
        :param directory: path to directory of data
        :param transform: makes transforms
        :param set name: set name (test, train, val)
        '''
        assert set_name in ['test', 'train', 'val'], "choose valid set name"

        self.img_dir = directory+"/"+set_name
        self.transforms = transforms
        self.set_name = set_name

        if set_name == 'test':
            self.set_list = constants.TEST_NAMES
        elif set_name == 'train':
            self.set_list = constants.TRAIN_NAMES
        elif set_name == 'val':
            self.set_list = constants.VAL_NAMES

    def __len__(self):
        return len(self.set_list)*1000

    def __getitem__(self, idx: int):
        img_label, index = divmod(idx, 1000) #returns the img label and file index 
        folder_name = self.set_list[img_label]
        label = {i:0 for i in range(10)}

        first_num = int(folder_name[0])
        second_num = int(folder_name[1])

        label = torch.zeros((10))
        label[first_num] += 1
        if label[second_num] == 0:
            label[second_num] += 1
        
        file_path = self.img_dir+'/'+folder_name+'/'+str(index)+'_'+folder_name+'.png'
        img = Image.open(file_path)
        img = img.convert('RGB')
        transforms = self.transforms
        if transforms is not None:
            img = transforms(img)
        img = torch.squeeze(img).permute(1,2,0)

        return img, label

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_data = OverlapMNIST(IMG_DIR, transform, 'train')
    plt.imshow(train_data[0][0], cmap = 'gray')
    plt.title(train_data[0][1])
    plt.show()