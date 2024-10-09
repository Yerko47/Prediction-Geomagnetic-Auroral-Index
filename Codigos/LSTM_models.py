import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

####### [ LSTM 1 ] #######
class LSTM_1(nn.Module):
    def __init__(self, input_size, drop, num_layer, device):
        super(LSTM_1, self).__init__()
        self.num_layer = num_layer
        self.hidden_size = 320
        self.device = device

        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layer, batch_first=True, dropout=drop)
        
        self.fc_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, 160),
            nn.Linear(160,160),
            nn.Linear(160,80),
            nn.Linear(80,20),
            nn.Linear(20,5),
            nn.Linear(5,1)
        ])   

        self.activation = nn.ReLU()

    def forward(self, x):
        h_0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size).to(self.device)
        c_0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size).to(self.device)


        out, _ = self.lstm(x, (h_0, c_0))

        out = out[:, -1, :]   

        for fc in self.fc_layers[:-1]:
            out = fc(out)
            out = self.activation(out)

        out = self.fc_layers[-1](out)  

        return out


####### [ LSTM 2 ] #######
class LSTM_2(nn.Module):
    def __init__(self, input_size, drop, num_layer):
        super(LSTM_2, self).__init__()
        self.num_layer = num_layer
        self.hidden_size = 320

        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layer, batch_first=True, dropout=drop)
        
        self.fc_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, 160),
            nn.Linear(160,160),
            nn.Linear(160,80),
            nn.Linear(80,20),
            nn.Linear(20,5),
            nn.Linear(5,1)
        ])  

        self.activation = nn.ReLU()

    def forward(self, x):
        h_0 = torch.zeros(self.num_layer, x.size(1), self.hidden_size)
        c_0 = torch.zeros(self.num_layer, x.size(1), self.hidden_size)

        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]  

        for fc in self.fc_layers[:-1]:
            out = fc(out)
            out = self.activation(out)

        out = self.fc_layers[-1](out)  

        return out