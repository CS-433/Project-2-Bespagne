# Imports
import torch
import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!
from sklearn.model_selection import KFold

import numpy as np
import os
import pandas as pd

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import torch.nn as nn

"""
(Hyper)parameters of our convolutional neural network found by tuning with Ray Tune (see further on)

"""
input_size = 3 * 100
num_classes = 3
learning_rate = 0.045421358413054946
batch_size = 10
num_epochs = 5

"""
Multi-Layer Perceptron
Class inherits from nn.Module, a pytorch neural network of choice, must inherit from base class nn.Module.
"""
class MLP(nn.Module):
    def __init__(self, input_size = input_size, num_classes = num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 10)
        self.fc4 = nn.Linear(10, num_classes)

    def forward(self, x):
        """
        Forward propagation function.
        
        self: self
        x: tensor of shape (1, input_size)
        
        returns: tensor of shape (1, num_classes) serving as a feature vector.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

"""
Set device cuda for GPU if it's available otherwise run on the CPU
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Load labels from csv file
"""
labels = np.genfromtxt('Data/Labels/avg_smooth.csv', delimiter=',')

"""
Make labels categorical: 3 quantification levels
"""
df_labels = pd.DataFrame(labels)
df_labels = pd.cut(df_labels[0],bins=[0, 0.013, 0.017, 0.03],labels=[0,1,2])
labels = df_labels.to_numpy()

"""
Load .npy files into one big array
"""
data = []
dir = 'Data/input_MLP/'
pac = np.load('Data/PAC_afterCNN.npy')
for filename in os.listdir(dir):
    sample = np.load(dir+filename)
    sample = np.append(sample, pac)
    data.append(sample.flatten())
data = np.array(data)

"""
Transform data to torch tensors
"""
tensor_x = torch.Tensor(data)
tensor_y = torch.Tensor(labels)
tensor_y = tensor_y.type(torch.LongTensor)

"""
Create dataset and data loader
"""
dataset = TensorDataset(tensor_x,tensor_y)

"""
Split into train and test sets
"""
test_size = int(0.3*len(dataset))
train_size = len(dataset) - test_size

train_data,test_data = random_split(dataset,[train_size,test_size])

train_loader = DataLoader(train_data, batch_size, shuffle = False, num_workers = 4, pin_memory = True)
test_loader = DataLoader(test_data, batch_size, shuffle = False, num_workers = 4, pin_memory = True)

"""
Initialize network
""" 
model = MLP(input_size, num_classes).to(device)
"""
Loss and optimizer
""" 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def reset_weights(m):
    """
    Try resetting model weights to avoid
    weight leakage.

    m: a PyTorch model

    return: None
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

"""
 Accuracy function.

 Check accuracy on training & test to see how good the model is
 
 loader: a pytorch dataloader
 model: a pytorch model
 
 returns: model accuracy as a numerical value
"""
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            #x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples

"""
Define the 5-fold Cross Validator
"""
k_folds = 5
kfold = KFold(n_splits=k_folds, shuffle=False)
train_acc = []
test_acc = []
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    print(f'Fold {fold}')
    model.apply(reset_weights)
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    trainloader = torch.utils.data.DataLoader(
                      dataset, batch_size=10, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      dataset, batch_size=10, sampler=test_subsampler)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for batch_idx, (input_, labels) in enumerate(tqdm(trainloader)):
            # Get data to cuda if possible
            input_ = input_.to(device=device)
            targets = labels.to(device=device)

            # forward
            scores = model(input_)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
        train_acc.append(check_accuracy(trainloader, model).item()*100)
        test_acc.append(check_accuracy(testloader, model).item()*100)
    print('Train Accuracy for fold %d: %d %%' % (fold, 100.0 * check_accuracy(trainloader, model)))
    print('Test Accuracy for fold %d: %d %%' % (fold, 100.0 * check_accuracy(testloader, model)))
print('Averaged Train Accuracy over %d k-folds: %d %%' % (k_folds, np.array(train_acc).mean()))
print('Averaged Test Accuracy over %d k-folds: %d %%' % (k_folds, np.array(test_acc).mean()))