import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import time
from constants import IMG_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR
import copy
from dataloader import OverlapMNIST
from tqdm import tqdm
import matplotlib.pyplot as plt

input_size = 224
batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_loop(model, dataloader, criterion):
    '''
    Tests the test accuracy and loss of the resnet dataset.
    '''
    since = time.time()
    
    model.eval()

    running_loss = 0.0
    running_corrects = 0.0

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        m = nn.Sigmoid()
        outputs = m(outputs)
        preds = torch.where(outputs > 1/2, 1, 0)

        running_loss += loss.item() * inputs.size(0)
                
        for i in range(batch_size):
            if torch.equal(preds[:,i], labels[:,i]):
                running_corrects += 1

    loss = running_loss/len(dataloader.dataset)
    acc = running_corrects/len(dataloader.dataset)

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(loss, acc))


if __name__ == '__main__':

    model = torch.load('./src/models/olmnist_resnet.pt').to(device)

    data_transforms = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = OverlapMNIST(IMG_DIR, data_transforms, 'test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = True) 
    
    criterion = nn.BCEWithLogitsLoss()
    test_loop(model, test_dataloader, criterion)