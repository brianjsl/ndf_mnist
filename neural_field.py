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

input_size = 224
batch_size = 32 
num_epochs = 15
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = torch.load('./src/models/olmnist_resnet.pt', map_location=device)

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
            for i, (inputs, labels) in tqdm(dataloaders[phase]):
                labels = labels.to(device).view(batch_size,-1)
                
                #zero the parameter gradients
                optimizer.zero_grad()

                #forward and track history if train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model((inputs[0].to(device), inputs[1].to(device)))
                    m = nn.Sigmoid()
                    outputs = 255*m(outputs)
                    loss = criterion(outputs, labels.float())
                
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            
                #statistics to keep track of
                running_loss += loss.item()
                
                running_corrects += torch.sum(outputs == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                    val_acc_history.append(epoch_acc)
        torch.save(model, './checkpoints/chkpt_{}.pt'.format(epoch))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

class NeuralField(nn.Module):
    def __init__(self):
        super(NeuralField, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12, 512), 
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,1)
        )
    
    def forward(self, x):
        z = encoder(x[0])
        coords = x[1].view(batch_size, 2)
        input = torch.cat((z,coords), 1)
        intensity = self.linear_relu_stack(input)
        return intensity

if __name__ == '__main__':
    model = NeuralField()
    model = nn.DataParallel(model)
    model = model.to(device)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print('Initializing Datasets and Dataloaders...')

    #create training and validation datasets
    image_datasets = {x: OverlapMNISTNDF(IMG_DIR, data_transforms[x], x) for x in ['train', 'val']}
    #create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = batch_size,
                        shuffle = True, num_workers = 2) for x in ['train', 'val']}
    print("Done Initializing Data.")

    #Initilaize Optimizer
    optimizer_ft =  optim.Adam(model.parameters(), lr = learning_rate)
    test_dataset = OverlapMNISTNDF(IMG_DIR, data_transforms, 'train')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = True)

    criterion = nn.MSELoss()
    model, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, 
                        num_epochs=num_epochs)
    
    torch.save(model, 'neural_field.py')


    
