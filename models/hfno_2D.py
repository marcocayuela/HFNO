#############################################
## Fourier parallel architecture without w ##
#############################################

#libraries
import os 
import sys 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import MLP


class HFNO2D_rect(nn.Module):
    def __init__(self, modes_layers_z, modes_x, depth, width_MLP, n_layers_MLP, input_size, output_size):
        super(HFNO2D_rect, self).__init__()

        """
        """

        self.modes_x = modes_x  # int: fixed number of modes used for the x direction 
        self.modes_layers_z = modes_layers_z  #list of int: determine the number of modes each layer take into account (ex: [4,8] --> layer 1 look at 4 first modes and layer 2 at 5th to 8th mode)
        self.depth = depth # size of the space where the input is projected 
        self.width_MLP = width_MLP  # width of the MLPs used for the transform in Fourier space
        self.n_layers_MLP = n_layers_MLP #nbr of layers for the MLPs
        self.input_size = input_size 
        self.output_size = output_size

        self.activation = nn.LeakyReLU()

        # Lifting transformation, corresponding to a simple linear transformation
        self.P = nn.Linear(self.input_size, self.depth)  # input channel is 2: (u(x,y),v(x,y))
        
        # n_l sequential layers
        self.sublayers = [] # will be composed of MLPs 
        for i in range(len(self.modes_layers_z)):
            if i==0:
                k_mode_z = self.modes_layers_z[i]
            else:
                k_mode_z = self.modes_layers_z[i]-self.modes_layers_z[i-1]
            layers_width = [2*self.depth*k_mode_z*(2*self.modes_x-1)]+self.n_layers_MLP*[self.width_MLP]+[2*k_mode_z*(2*self.modes_x-1)*self.output_size]

            self.sublayers.append(MLP.FFNN(layers_width))

            self.sublayers = nn.ModuleList(self.sublayers)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, indexes=None):

        mx = x.shape[1]
        mz = x.shape[2]
        x = self.P(x)
        x = x.permute(0, 3, 1, 2)
        ft = torch.fft.rfft2(x)
        fourier_parts = []

        for i in range(len(self.modes_layers_z)):
            if i==0:
                k_start = 0
                k_end = self.modes_layers_z[0]
            else:
                k_start = self.modes_layers_z[i-1]
                k_end = self.modes_layers_z[i]
                
            #besoin d'une fonction qui crée l'input 
            
            input_sub = torch.concatenate([ft[...,:self.modes_x,k_start:k_end],ft[...,(-self.modes_x+1):,k_start:k_end]], dim=-2) 
            input_sub = torch.concatenate((input_sub.real, input_sub.imag), dim=-1)
            sh = input_sub.shape
            input_sub = input_sub.reshape(sh[0],sh[1]*sh[2]*sh[3])

            output_sub = self.sublayers[i](input_sub)
            output_sub = output_sub.reshape(sh[0],self.output_size,sh[2],sh[3]//2,2)
            output_sub = output_sub[...,0] + 1j*output_sub[...,1]

            output_pad = torch.zeros(sh[0],self.output_size, mx, mz//2+1, dtype=torch.cfloat)
            output_pad[:,:,:self.modes_x,k_start:k_end] = output_sub[:,:,:self.modes_x]
            output_pad[:,:,(-self.modes_x):,k_start:k_end] = output_sub[:,:,(-self.modes_x):]
            
            ft_out = torch.fft.irfft2(output_pad, s=(mx, mz)).permute(0,2,3,1)
            fourier_parts.append(ft_out)

        if indexes is None:
            indexes = np.arange(len(fourier_parts))
        elif isinstance(indexes, int):
            indexes = np.arange(indexes)
        else:
            indexes = np.array(indexes)            

        if indexes.shape[0]==0:
            z = torch.zeros_like(torch.fft.irfft2(output_pad,s=(mx, mz)))
            z = z.permute(0,2,3,1)
        else:
            z = torch.sum(torch.stack([fourier_parts[i] for i in indexes]), axis=0)

        return z
    
    

class HFNO2D_rect_w(nn.Module):
    def __init__(self, modes_layers_z, modes_x, depth, width_MLP, n_layers_MLP, input_size, output_size, type_w="linear"):
        super(HFNO2D_rect_w, self).__init__()

        """
        """

        self.modes_x = modes_x  # int: fixed number of modes used for the x direction 
        self.modes_layers_z = modes_layers_z  #list of int: determine the number of modes each layer take into account (ex: [4,8] --> layer 1 look at 4 first modes and layer 2 at 5th to 8th mode)
        self.depth = depth # size of the space where the input is projected 
        self.width_MLP = width_MLP  # width of the MLPs used for the transform in Fourier space
        self.n_layers_MLP = n_layers_MLP #nbr of layers for the MLPs
        self.input_size = input_size 
        self.output_size = output_size

        self.activation = nn.LeakyReLU()

        # Lifting transformation, corresponding to a simple linear transformation
        self.P = nn.Linear(self.input_size, self.depth)  # input channel is 2: (u(x,y),v(x,y))
        
        # n_l sequential layers
        self.sublayers = [] # will be composed of MLPs 
        for i in range(len(self.modes_layers_z)):
            if i==0:
                k_mode_z = self.modes_layers_z[i]
            else:
                k_mode_z = self.modes_layers_z[i]-self.modes_layers_z[i-1]
            layers_width = [2*self.depth*k_mode_z*(2*self.modes_x-1)]+self.n_layers_MLP*[self.width_MLP]+[2*k_mode_z*(2*self.modes_x-1)*self.output_size]

            self.sublayers.append(MLP.FFNN(layers_width))

            self.sublayers = nn.ModuleList(self.sublayers)

        
        default_cnn = nn.Sequential(
            nn.Conv2d(self.depth, 32, 3, padding="same"),
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.Conv2d(64, 32, 3, padding="same"),
            nn.Conv2d(32, 8, 3, padding="same"),
            nn.Conv2d(8, self.output_size, 3, padding="same")
        )

        # Use custom CNN if provided, otherwise use the default CNN
        self.W = nn.Conv2d(self.depth, self.output_size, 1, padding='same') if type_w=="linear" else default_cnn

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, indexes=None, add_w=True):

        mx = x.shape[1]
        mz = x.shape[2]
        x = self.P(x)
        x = x.permute(0, 3, 1, 2)
        ft = torch.fft.rfft2(x)
        fourier_parts = []

        for i in range(len(self.modes_layers_z)):
            if i==0:
                k_start = 0
                k_end = self.modes_layers_z[0]
            else:
                k_start = self.modes_layers_z[i-1]
                k_end = self.modes_layers_z[i]
                
            #besoin d'une fonction qui crée l'input 
            
            input_sub = torch.concatenate([ft[...,:self.modes_x,k_start:k_end],ft[...,(-self.modes_x+1):,k_start:k_end]], dim=-2) 
            input_sub = torch.concatenate((input_sub.real, input_sub.imag), dim=-1)
            sh = input_sub.shape
            input_sub = input_sub.reshape(sh[0],sh[1]*sh[2]*sh[3])

            output_sub = self.sublayers[i](input_sub)
            output_sub = output_sub.reshape(sh[0],self.output_size,sh[2],sh[3]//2,2)
            output_sub = output_sub[...,0] + 1j*output_sub[...,1]

            output_pad = torch.zeros(sh[0],self.output_size, mx, mz//2+1, dtype=torch.cfloat)
            output_pad[:,:,:self.modes_x,k_start:k_end] = output_sub[:,:,:self.modes_x]
            output_pad[:,:,(-self.modes_x):,k_start:k_end] = output_sub[:,:,(-self.modes_x):]
            
            ft_out = torch.fft.irfft2(output_pad, s=(mx, mz)).permute(0,2,3,1)
            fourier_parts.append(ft_out)

        if indexes is None:
            indexes = np.arange(len(fourier_parts))
        elif isinstance(indexes, int):
            indexes = np.arange(indexes)
        else:
            indexes = np.array(indexes)            

        if indexes.shape[0]==0:
            z = torch.zeros_like(torch.fft.irfft2(output_pad,s=(mx, mz)))
            z = z.permute(0,2,3,1)
        else:
            z = torch.sum(torch.stack([fourier_parts[i] for i in indexes]), axis=0)

        if add_w:
            linear_part = self.W(fft_out).permute(0, 2, 3, 1)
            z += linear_part 
        return z
    

    
    




class HFNO_2D(nn.Module):
    def __init__(self,modes, depth, width_MLP, n_layers_MLP, input_size, output_size, res):
        super(HFNO_2D, self).__init__()

        """
        """

        self.modes = modes  # int: fixed number of modes used for the x direction 
        self.depth = depth # size of the space where the input is projected 
        self.width_MLP = width_MLP  # width of the MLPs used for the transform in Fourier space
        self.n_layers_MLP = n_layers_MLP #nbr of layers for the MLPs
        self.input_size = input_size 
        self.output_size = output_size
        self.res = res

        self.activation = nn.LeakyReLU()

        # Lifting transformation, corresponding to a simple linear transformation
        self.P = nn.Linear(self.input_size, self.depth)  # input channel is 2: (u(x,y),v(x,y))
        
        kxx = np.fft.fftfreq(self.res, d=1.0)*self.res
        kyy = np.fft.fftfreq(self.res, d=1.0)*self.res
        kx, ky = np.meshgrid(kxx, kyy, indexing='ij')
        self.k_magnitude = np.sqrt(kx**2 + ky**2) 
        self.rk_magnitude = self.k_magnitude[:,:self.res//2+1]

        self.mask_list = []
        # n_l sequential layers
        self.sublayers = [] # will be composed of MLPs 
        for i in range(len(self.modes)):
            if i==0:
                k_end = self.modes[i]
                k_start = 0
            else:
                k_end = self.modes[i]
                k_start = self.modes[i-1]

            mask = np.logical_and((self.rk_magnitude <= k_end), self.rk_magnitude >= k_start)
            self.mask_list.append(mask)

            input_modes = mask.sum()
            layers_width = [2*self.depth*input_modes]+self.n_layers_MLP*[self.width_MLP]+[2*input_modes*self.output_size]

            self.sublayers.append(MLP.FFNN(layers_width))

            self.sublayers = nn.ModuleList(self.sublayers)


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, indexes=None):

        mx = x.shape[1]
        mz = x.shape[2]
        x = self.P(x)
        x = x.permute(0, 3, 1, 2)
        fft_x = torch.fft.rfft2(x)
        fourier_parts = []

        for i in range(len(self.modes)):
            
            mask = self.mask_list[i]
            input_sub = fft_x[:,:,mask] 
            input_sub = torch.concatenate((input_sub.real, input_sub.imag), dim=-1)
            sh = input_sub.shape
            input_sub = input_sub.reshape(sh[0],sh[1]*sh[2])

            output_sub = self.sublayers[i](input_sub)
            output_sub = output_sub.reshape(sh[0],self.output_size, sh[2]//2,2)
            output_sub = output_sub[...,0] + 1j*output_sub[...,1]

            output_pad = torch.zeros(sh[0],self.output_size, self.res, self.res//2+1, dtype=torch.cfloat)
            output_pad[:,:,mask] = output_sub[:,:]
            
            ft_out = torch.fft.irfft2(output_pad, s=(mx, mz)).permute(0,2,3,1)
            fourier_parts.append(ft_out)

        if indexes is None:
            indexes = np.arange(len(fourier_parts))
        elif isinstance(indexes, int):
            indexes = np.arange(indexes)
        else:
            indexes = np.array(indexes)            

        if indexes.shape[0]==0:
            z = torch.zeros_like(torch.fft.irfft2(output_pad,s=(mx, mz)))
            z = z.permute(0,2,3,1)
        else:
            z = torch.sum(torch.stack([fourier_parts[i] for i in indexes]), axis=0)

        return z
    

class HFNO2D_l2_overlap(nn.Module):
    def __init__(self,modes_start, modes_end, depth, width_MLP, n_layers_MLP, input_size, output_size, res):
        super(HFNO2D_l2_overlap, self).__init__()

        """
        """

        self.modes_start = [0]+ modes_start
        self.modes_end = modes_end  # int: fixed number of modes used for the x direction 
        self.depth = depth # size of the space where the input is projected 
        self.width_MLP = width_MLP  # width of the MLPs used for the transform in Fourier space
        self.n_layers_MLP = n_layers_MLP #nbr of layers for the MLPs
        self.input_size = input_size 
        self.output_size = output_size
        self.res = res

        self.activation = nn.LeakyReLU()

        # Lifting transformation, corresponding to a simple linear transformation
        self.P = nn.Linear(self.input_size, self.depth)  # input channel is 2: (u(x,y),v(x,y))
        
        kxx = np.fft.fftfreq(self.res, d=1.0)*self.res
        kyy = np.fft.fftfreq(self.res, d=1.0)*self.res
        kx, ky = np.meshgrid(kxx, kyy, indexing='ij')
        self.k_magnitude = np.sqrt(kx**2 + ky**2) 
        self.rk_magnitude = self.k_magnitude[:,:self.res//2+1]

        self.mask_list = []
        # n_l sequential layers
        self.sublayers = [] # will be composed of MLPs 
        for i in range(len(self.modes_start)):
            k_end = self.modes_end[i]
            k_start = self.modes_start[i]

            mask = np.logical_and((self.rk_magnitude <= k_end), self.rk_magnitude >= k_start)
            self.mask_list.append(mask)

            input_modes = mask.sum()
            layers_width = [2*self.depth*input_modes]+self.n_layers_MLP*[self.width_MLP]+[2*input_modes*self.output_size]

            self.sublayers.append(MLP.FFNN(layers_width))

            self.sublayers = nn.ModuleList(self.sublayers)


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, indexes=None):

        mx = x.shape[1]
        mz = x.shape[2]
        x = self.P(x)
        x = x.permute(0, 3, 1, 2)
        fft_x = torch.fft.rfft2(x)
        fourier_parts = []

        for i in range(len(self.modes_start)):
            
            mask = self.mask_list[i]
            input_sub = fft_x[:,:,mask] 
            input_sub = torch.concatenate((input_sub.real, input_sub.imag), dim=-1)
            sh = input_sub.shape
            input_sub = input_sub.reshape(sh[0],sh[1]*sh[2])

            output_sub = self.sublayers[i](input_sub)
            output_sub = output_sub.reshape(sh[0],self.output_size, sh[2]//2,2)
            output_sub = output_sub[...,0] + 1j*output_sub[...,1]

            output_pad = torch.zeros(sh[0],self.output_size, self.res, self.res//2+1, dtype=torch.cfloat)
            output_pad[:,:,mask] = output_sub[:,:]
            
            ft_out = torch.fft.irfft2(output_pad, s=(mx, mz)).permute(0,2,3,1)
            fourier_parts.append(ft_out)

        if indexes is None:
            indexes = np.arange(len(fourier_parts))
        elif isinstance(indexes, int):
            indexes = np.arange(indexes)
        else:
            indexes = np.array(indexes)            

        if indexes.shape[0]==0:
            z = torch.zeros_like(torch.fft.irfft2(output_pad,s=(mx, mz)))
            z = z.permute(0,2,3,1)
        else:
            z = torch.sum(torch.stack([fourier_parts[i] for i in indexes]), axis=0)

        return z
    


class FNO2D_v2_CNN(nn.Module):
    def __init__(self, modes, depth, width_MLP, n_layers_MLP, input_size, output_size, res, last_modes_cutoff, custom_cnn=None):
        super(FNO2D_v2_CNN, self).__init__()

        """
        FNO2D_v2_CNN architecture with optional custom CNN.
        If `custom_cnn` is provided, it will replace the default CNN.
        """

        self.modes = modes  # int: fixed number of modes used for the x direction 
        self.depth = depth  # size of the space where the input is projected 
        self.width_MLP = width_MLP  # width of the MLPs used for the transform in Fourier space
        self.n_layers_MLP = n_layers_MLP  # number of layers for the MLPs
        self.input_size = input_size 
        self.output_size = output_size
        self.res = res
        self.last_modes_cutoff = last_modes_cutoff
        self.activation = nn.LeakyReLU()

        # Lifting transformation, corresponding to a simple linear transformation
        self.P = nn.Linear(self.input_size, self.depth)  # input channel is 2: (u(x,y),v(x,y))
        
        kxx = np.fft.fftfreq(self.res, d=1.0)*self.res
        kyy = np.fft.fftfreq(self.res, d=1.0)*self.res
        kx, ky = np.meshgrid(kxx, kyy, indexing='ij')
        self.k_magnitude = np.sqrt(kx**2 + ky**2) 
        self.rk_magnitude = self.k_magnitude[:,:self.res//2+1]

        self.mask_list = []
        # n_l sequential layers
        self.sublayers = []  # will be composed of MLPs 
        for i in range(len(self.modes)):
            if i == 0:
                k_end = self.modes[i]
                k_start = 0
            else:
                k_end = self.modes[i]
                k_start = self.modes[i-1]

            mask = np.logical_and((self.rk_magnitude <= k_end), self.rk_magnitude >= k_start)
            self.mask_list.append(mask)

            input_modes = mask.sum()
            layers_width = [2*self.depth*input_modes] + self.n_layers_MLP*[self.width_MLP] + [2*input_modes*self.output_size]

            self.sublayers.append(MLP.FFNN(layers_width))

            self.sublayers = nn.ModuleList(self.sublayers)

        # Default CNN architecture
        default_cnn = nn.Sequential(
            nn.Conv2d(self.depth, 32, 3, padding="same"),
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.Conv2d(64, 32, 3, padding="same"),
            nn.Conv2d(32, 8, 3, padding="same"),
            nn.Conv2d(8, self.output_size, 3, padding="same")
        )

        # Use custom CNN if provided, otherwise use the default CNN
        self.W = custom_cnn if custom_cnn is not None else default_cnn

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, indexes=None, add_w=True):

        mx = x.shape[1]
        mz = x.shape[2]
        x = self.P(x)
        x = x.permute(0, 3, 1, 2)
        fft_x = torch.fft.rfft2(x)
        fourier_parts = []

        for i in range(len(self.modes)):
            
            mask = self.mask_list[i]
            input_sub = fft_x[:,:,mask] 
            input_sub = torch.concatenate((input_sub.real, input_sub.imag), dim=-1)
            sh = input_sub.shape
            input_sub = input_sub.reshape(sh[0], sh[1]*sh[2])

            output_sub = self.sublayers[i](input_sub)
            output_sub = output_sub.reshape(sh[0], self.output_size, sh[2]//2, 2)
            output_sub = output_sub[..., 0] + 1j*output_sub[..., 1]

            output_pad = torch.zeros(sh[0], self.output_size, self.res, self.res//2+1, dtype=torch.cfloat)
            output_pad[:,:,mask] = output_sub[:,:]
            
            ft_out = torch.fft.irfft2(output_pad, s=(mx, mz)).permute(0, 2, 3, 1)
            fourier_parts.append(ft_out)

        if indexes is None:
            indexes = np.arange(len(fourier_parts))
        elif isinstance(indexes, int):
            indexes = np.arange(indexes)
        else:
            indexes = np.array(indexes)            

        if indexes.shape[0] == 0:
            z = torch.zeros_like(torch.fft.irfft2(output_pad, s=(mx, mz)))
            z = z.permute(0, 2, 3, 1)
        else:
            z = torch.sum(torch.stack([fourier_parts[i] for i in indexes]), axis=0)

        fft_w_pad = torch.zeros_like(fft_x)
        mask = np.logical_and((self.rk_magnitude <= self.last_modes_cutoff), self.rk_magnitude >= self.modes[-1])
        fft_w_pad[:,:,mask] = fft_x[:,:,mask]
        fft_out = torch.fft.irfft2(fft_w_pad, s=(mx, mz))

        if add_w:
            linear_part = self.W(fft_out).permute(0, 2, 3, 1)
            z += linear_part 

        return z