import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ANN_1(nn.Module):
    def __init__(self, input_size, drop):
        super(ANN_1, self).__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(input_size,320),
            nn.Linear(320,160),
            nn.Linear(160,160),
            nn.Linear(160,80),
            nn.Linear(80,20),
            nn.Linear(20,5),
            nn.Linear(5,1)
        ])        

        self.drop_layers = nn.ModuleList([
            nn.Dropout(drop),
            nn.Dropout(drop),
            nn.Dropout(drop),
            nn.Dropout(drop)
        ])

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(320),
            nn.BatchNorm1d(160),
            nn.BatchNorm1d(160),
            nn.BatchNorm1d(80),
            nn.BatchNorm1d(20),
            nn.BatchNorm1d(5),
        ])

        self.activation = nn.ReLU()

    def forward(self, x):
        for i, (fc, bn) in enumerate(zip(self.fc_layers[:-1], self.bn_layers)):
            x = fc(x)
            x = bn(x)
            x = self.activation(x)
            if i < len(self.drop_layers):
                x = self.drop_layers[i](x)

        x = self.fc_layers[-1](x) 
        return x

class ANN_2(nn.Module):
    def __init__(self, input_size, drop):
        super(ANN_2, self).__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(input_size,320),
            nn.Linear(320,160),
            nn.Linear(160,20),
            nn.Linear(20,10),
            nn.Linear(10,5),
            nn.Linear(5,1)
        ])        

        self.drop_layers = nn.ModuleList([
            nn.Dropout(drop),
            nn.Dropout(drop),
            nn.Dropout(drop),
            nn.Dropout(drop)
        ])

        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(320),
            nn.BatchNorm1d(160),
            nn.BatchNorm1d(20),
            nn.BatchNorm1d(10),
            nn.BatchNorm1d(5),
        ])

        self.activation = nn.ReLU()

    def forward(self, x):
        for i, (fc, bn) in enumerate(zip(self.fc_layers[:-1], self.bn_layers)):
            x = fc(x)
            x = bn(x)
            x = self.activation(x)
            if i < len(self.drop_layers):
                x = self.drop_layers[i](x)

        x = self.fc_layers[-1](x) 
        return x
