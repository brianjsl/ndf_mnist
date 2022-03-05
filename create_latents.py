import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import time
from constants import IMG_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR
import copy
from tqdm import tqdm

data_dir = IMG_DIR
num_classes = 10
batch_size = 8
num_epochs = 15

#Push to GPU 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_model(num_classes, use_pretrained = True):
    '''
    Initializes a ResNet model with a given number of classes. 
    
    params:
    @num_classes: defines number of classes to use for the model
    @use_pretrained: defines whether or not to use pretrained weights
    in training phase. Defaults to True.
    '''
    #fine-tuned model
    model_ft = models.resnet18(pretrained=use_pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model_ft, input_size
    
    

def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    '''
    Trains the model for given number of epochs.

    params:
    @ model: model to train
    @ dataloaders: dictionary of dataloaders
    @ criterion: loss function/criterion
    @ optimizer: optimizer used 
    @ num_epochs: number of epochs to train for
    '''
    since = time.time()

    #validation accuracy history
    val_acc_history = []
    
    #weights of best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('_'*10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

        #keep track of losses and corrects        
        running_loss = 0.0
        running_corrects = 0.0

        #Iterate over data
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            #zero the parameter gradients
            optimizer.zero_grad()

            #forward and track history if train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                #map to an output
                outputs = nn.Sigmoid(outputs)
                preds = torch.where(outputs > 1/2, 1, 0)
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
            
            #statistics to keep track of
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
            
if __name__ == '__main__':
    
        



