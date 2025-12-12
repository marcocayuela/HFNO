import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNN(nn.Module):

    """Feed Forward Neural Network 
    Args:
        layers_width (list): list of integers representing the width of each layer;
        activation (nn.Module, optional): activation function to use between layers. Defaults to nn.GELU();
        device (str, optional): device to run the model on. Defaults to "cpu".
    Returns:
        nn.Module: Feed Forward Neural Network model.
    """   

    def __init__(self, layers_width, activation=nn.GELU(), device="cpu"):
        super(FFNN, self).__init__()

        self.layers_width = layers_width
        self.depth = len(self.layers_width)
        self.activation = activation

        self.model = nn.Sequential()
        for i in range(self.depth - 2):
            self.model.append(nn.Linear(self.layers_width[i], self.layers_width[i+1], device=device))
            self.model.append(self.activation)
        self.model.append(nn.Linear(self.layers_width[-2], self.layers_width[-1], device=device))

    def forward(self, input):
        return self.model(input)