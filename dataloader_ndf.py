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

class OverlapMNISTNDF(Dataset):
    '''
    Dataset of Overlapping MNIST Images for use in NDFs. 
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
        if self.set_name == 'train':
            return len(self.set_list)*100*1024
        else:
            return len(self.set_list)*10*1024

    def __getitem__(self, idx: int):
        if self.set_name == 'train':
            img_label, rem = divmod(idx, 1024*100) #returns the img label and file index
        else:
            img_label, rem = divmod(idx, 1024*10) #returns the img label and file index 
        folder_name = self.set_list[img_label]

        index, pos = divmod(rem,1024) #index of image
 
        file_path = self.img_dir+'/'+folder_name+'/'+str(index)+'_'+folder_name+'.png'
        
        img = Image.open(file_path)
        img = img.convert('RGB')
        x_coord, y_coord = divmod(pos, 32) #tuple used in NDF
        pos = torch.tensor([x_coord,y_coord]).reshape(2,-1)

        intensity = torch.tensor(img.getpixel((x_coord,y_coord)))

        transforms = self.transforms
        if transforms is not None:
            img = transforms(img)
 
        return ((img, pos), intensity[0])

if __name__ == '__main__':
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data = OverlapMNISTNDF(IMG_DIR, data_transforms['train'],'train')
    print(data[0])
