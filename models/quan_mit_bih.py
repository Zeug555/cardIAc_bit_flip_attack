import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

from .quantization import *


class cnn_bones(nn.Module):

    def __init__(self, num_classes):
        super(cnn_bones, self).__init__()
        
        # self.conv1 = quan_Conv1d(in_channels=1, out_channels=8, kernel_size=3)
        # self.pool = nn.MaxPool1d(kernel_size=2)
        # self.flatten = nn.Flatten()
        # self.fc1 = quan_Linear(8 * ((180 - 2) // 2), 64)
        # self.dropout = nn.Dropout(0.25)
        # self.fc2 = quan_Linear(64, num_classes)

        # Première couche Conv1D : (input_shape=(180, 1), filters=4, kernel_size=21)
        self.conv1 = quan_Conv1d(in_channels=1, out_channels=4, kernel_size=21)
        # MaxPooling1D avec pool_size=3, stride=3
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)

        # Deuxième couche Conv1D : (filters=4, kernel_size=21)
        self.conv2 = quan_Conv1d(in_channels=4, out_channels=4, kernel_size=21)
        # MaxPooling1D avec pool_size=3, stride=3
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)

        # Flatten pour préparer l'entrée du Fully Connected
        # Automatique via reshape ou nn.Flatten
        self.flatten = nn.Flatten()

        # Couche dense (Dense 32 unités, activation='relu')
        # Taille d'entrée calculée : 29 * 4 = 116
        self.fc1 = quan_Linear(44, 32)

        # Couche dense finale : (5 unités pour num_classes)
        self.fc2 = quan_Linear(32, num_classes)


    def forward(self, x):
        # x = self.pool(torch.relu(self.conv1(x)))
        # x = self.flatten(x)
        # x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = self.fc2(x)
        # return x
    
        # x shape : (batch_size, 1, 181)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        # Après la première convolution et pooling
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # Après la deuxième convolution et pooling
        x = self.flatten(x)  # Flatten
        x = F.relu(self.fc1(x))  # Fully Connected Layer
        x = self.fc2(x)  # Output Layer
        return x
        
def cnn_quan(num_classes=5):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
  Args:
    num_classes (uint): number of classes
  """
    model = cnn_bones(num_classes)
    return model