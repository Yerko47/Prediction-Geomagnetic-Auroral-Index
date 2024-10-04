import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class CNN_1(nn.Module):
    def __init__(self, input_size):
        self.conv1 = nn.Conv1d(input_size)