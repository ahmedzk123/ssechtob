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


# v2: side to move input (768 pieces + 4 castling), nn learns a correction on top of exact material
# piece slots per square: 0-5 = my P N B R Q K, 6-11 = opp P N B R Q K

class ANN2(nn.Module):
    def __init__(self):
        super(ANN2, self).__init__()
        self.W0 = nn.Linear(772, 512)
        self.W1 = nn.Linear(512, 256)
        self.W2 = nn.Linear(256, 128)
        self.W3 = nn.Linear(128, 64)
        self.W4 = nn.Linear(64, 1)

        # start as pure material: tanh(material / 4)
        nn.init.zeros_(self.W4.weight)
        nn.init.zeros_(self.W4.bias)
        self.a = nn.Parameter(torch.tensor(1.0))

        values = torch.zeros(772)
        for square in range(64):
            for p, v in enumerate([1, 3, 3, 5, 9, 0]):
                values[square * 12 + p] = v
                values[square * 12 + 6 + p] = -v
        self.register_buffer("piece_values", values)

    def forward(self, x):
        material = x @ self.piece_values
        x0 = F.leaky_relu(self.W0(x))
        x1 = F.leaky_relu(self.W1(x0))
        x2 = F.leaky_relu(self.W2(x1))
        x3 = F.leaky_relu(self.W3(x2))
        y_hat = torch.tanh(self.W4(x3).squeeze(-1) + self.a * material / 4)
        return y_hat.unsqueeze(-1)
