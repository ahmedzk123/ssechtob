import torch
import torch.nn as nn 
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.W0 = nn.Linear(769, 512)
        self.W1 = nn.Linear(512, 256) 
        self.W2 = nn.Linear(256, 128)
        self.W3 = nn.Linear(128, 64)
        self.W4 = nn.Linear(64, 1)

    def forward(self, x):
        x0 = F.leaky_relu(self.W0(x))
        x1 = F.leaky_relu(self.W1(x0))
        x2 = F.leaky_relu(self.W2(x1))
        x3 = F.leaky_relu(self.W3(x2))
        y_hat = torch.tanh(self.W4(x3))
        return y_hat
