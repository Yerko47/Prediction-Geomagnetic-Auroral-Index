import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


### [ LSTM 1 ] ###
class LSTM_1(nn.Module):
    def __init__(self, input_size, drop, num_layer, device):
        super(LSTM_1, self).__init__()
        self.num_layer = num_layer

        self.lstm = nn.LSTM(input_size, 160, num_layers=num_layer, batch_first=True, dropout=drop)
        
        self.fc1 = nn.Linear(160, 80)
        self.fc2 = nn.Linear(80, 40)
        self.fc3 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(20, 5)
        self.fc5 = nn.Linear(5, 1)

        self.dropout = nn.Dropout(drop)
        self.relu = nn.ReLU()

    def forward(self, x):
        h_0 = torch.zeros(self.num_layer, x.size(0), 160, device=device)
        c_0 = torch.zeros(self.num_layer, x.size(0), 160, device=device)
        
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :] 
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc4(out)
        out = self.relu(out)
        
        out = self.fc5(out)

        return out
    

### [ LSTM 2 ] ###
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
        out = out[:, -1, :]  # Selecciona la última salida de la secuencia

        for fc in self.fc_layers[:-1]:
            out = fc(out)
            out = self.activation(out)
        out = self.fc_layers[-1](out)  # Última capa sin activación

        return out