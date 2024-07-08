import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


###### [ ANN Model ] ######
class ANN(nn.Module):
    def __init__(self, input_size, drop):
        super(ANN, self).__init__()
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        self.drop3 = nn.Dropout(drop)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

        self.fc1 = nn.Linear(input_size, 320)
        self.fc2 = nn.Linear(320, 160)
        self.fc3 = nn.Linear(160, 160)
        self.fc4 = nn.Linear(160, 80)
        self.fc5 = nn.Linear(80, 20)
        self.fc6 = nn.Linear(20, 5)
        self.fc7 = nn.Linear(5, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(self.drop1(self.fc2(out)))
        out = self.relu2(self.drop2(self.fc3(out)))
        out = self.relu3(self.drop3(self.fc4(out)))
        out = self.relu4(self.fc5(out))
        out = self.fc6(out)
        out = self.fc7(out)

        return out
    

###### [ LSTM Model ] ######
class LSTM(nn.Module):
    def __init__(self, input_size, drop, num_layer, device):
        super(LSTM, self).__init__()
        self.device = device
        self.num_layer = num_layer

        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        self.drop3 = nn.Dropout(drop)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

        self.lstm = nn.LSTM(input_size, 160, num_layers=num_layer, batch_first=True, dropout=drop)
        self.fc1 = nn.Linear(160, 80)
        self.fc2 = nn.Linear(80, 80)
        self.fc3 = nn.Linear(80, 40)
        self.fc4 = nn.Linear(40, 20)
        self.fc5 = nn.Linear(20, 5)
        self.fc6 = nn.Linear(5, 1)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layer, x.size(0), 320).to(self.device)
        c_0 = torch.zeros(self.num_layer, x.size(0), 320).to(self.device)
        out,_ = self.lstm(x, (h_0, c_0))
        out = out[:,-1,:]
        

        out = self.relu1(self.drop1(self.fc1(out)))
        out = self.relu2(self.drop2(self.fc2(out)))
        out = self.relu3(self.drop3(self.fc3(out)))
        out = self.relu4(self.fc4(out))
        out = self.fc5(out)
        out = self.fc6(out)

        return out