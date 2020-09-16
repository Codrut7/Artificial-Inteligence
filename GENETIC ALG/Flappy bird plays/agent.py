import torch
import torch.nn as nn
import numpy as np

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()

        # Define agenet hyper-params
        self.inp_size = 2
        self.hidden_size = 5
        self.out_size = 1

        # Activation
        #self.leaky_relu = torch.sigmoid()

        self.layer1 = nn.Linear(self.inp_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer3 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):

        x = self.layer1(x)
        #x = self.leaky_relu(x)
        x = self.layer2(x)
        #x = self.leaky_relu(x)
        x = self.layer3(x)
        #x = self.leaky_relu(x)

        x = torch.sigmoid(x)

        return x
