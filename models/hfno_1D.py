import torch
import torch.nn as nn
import numpy as np

import sys
import os
from models import MLP

### 

class HFNO_1D(nn.Module):
    def __init__(self, layer_modes, depth, width_MLP, n_layers_MLP, input_size, output_size, linear_W=True):
        super(HFNO_1D, self).__init__()

        self.layer_modes = layer_modes 
        self.depth = depth     
        self.width_MLP = width_MLP
        self.n_layers_MLP = n_layers_MLP
        self.input_size = input_size
        self.output_size = output_size
        self.k_modes = self.layer_modes[-1]
        self.n_sublayer = len(self.layer_modes)
        self.linear_W = linear_W
        
        self.activation = nn.GELU()

        self.P = nn.Linear(self.input_size, self.depth)

        if self.linear_W:
            self.W = nn.Conv1d(self.depth, 1, 1)

        
        self.sublayers = []
        for i in range(len(self.layer_modes)):
            if i==0:
                k_mode = self.layer_modes[i]
            else:
                k_mode = self.layer_modes[i]-self.layer_modes[i-1]
            layers_width = [2*self.depth*k_mode]+self.n_layers_MLP*[self.width_MLP]+[2*k_mode]
            self.sublayers.append(MLP.FFNN(layers_width))

        self.sublayers = nn.ModuleList(self.sublayers)
        

    def forward(self, x, indexes=None, sum_w=True):

        
        self.layer_activations = {}

        m = x.shape[1]
        x = self.P(x.unsqueeze(dim=-1))
        x = x.permute(0, 2, 1)
        ft = torch.fft.rfft(x)


        fourier_parts = []
        for i in range(len(self.layer_modes)):
            if i==0:
                k_start = 0
                k_end = self.layer_modes[0]
            else:
                k_start = self.layer_modes[i-1]
                k_end = self.layer_modes[i]
                

            len_layer = k_end-k_start
            input_sub = ft[...,k_start:k_end]
            input_sub = torch.concatenate((input_sub.real, input_sub.imag), dim=-1)

            sh = input_sub.shape
            input_sub = input_sub.reshape(sh[0],sh[1]*sh[2])
            output_sub = self.sublayers[i](input_sub)
            output_sub = output_sub[...,:len_layer] + 1j*output_sub[...,len_layer:]
            output_pad = torch.zeros(sh[0],self.k_modes, dtype=torch.cfloat)
            output_pad[:,k_start:k_end] = output_sub
            
            ft_out = torch.fft.irfft(output_pad, n=m)
        
            fourier_parts.append(ft_out)

        if indexes is None:
            indexes = np.arange(len(fourier_parts))
        elif isinstance(indexes, int):
            indexes = np.arange(indexes)
        else:
            indexes = np.array(indexes)            

        if indexes.shape[0]==0:
            z = torch.zeros_like(torch.fft.irfft(output_pad, n=m))
        else:
            z = torch.sum(torch.stack([fourier_parts[i] for i in indexes]), axis=0)
        if self.linear_W and sum_w:
            ft_w_part = torch.fft.rfft(x)
            ft_w_part[...,:self.layer_modes[-1]] = 0
            x_w_part = torch.fft.irfft(ft_w_part, n=m)
            linear_part = self.W(x_w_part).squeeze()
            z += linear_part 

        return z
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)