import torch
import torch.nn as nn
import torch.nn.functional as F



class IntegralKernel2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super(IntegralKernel2D, self).__init__()
        """
        2D Fourier layer. It computes FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels  # This is d_v
        self.out_channels = out_channels  # This is d_v
        self.modes_x = modes_x  # This is k_max
        self.modes_y = modes_y  # This is k_max
        
        self.scale = (1 / (in_channels * out_channels))

        # Parametrization R of the kernel
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes_x, self.modes_y, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]  # Batch size
        dv = x.shape[1]  # Lifting-dimension
        n = x.shape[-1]  # Number of grid points where input and intermediate states are evaluated

        out_ft = torch.zeros(batchsize, self.out_channels, self.modes_x, self.modes_y, device=x.device, dtype=torch.cfloat)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)
        # Use compl_mul1d to perform the multiplication between the relevant Fourier Modes and self.weights and fill the tensor out_tf with the corresponding values
        out_ft[:, :, :self.modes_x, :self.modes_y] = self.compl_mul2d(x_ft[:,:,:self.modes_x, :self.modes_y], self.weights)
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    



class FourierLayer2D(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y, l):
        super(FourierLayer2D, self).__init__()

        self.int_kern = IntegralKernel2D(in_channels, out_channels, modes_x, modes_y)
        self.w = nn.Conv2d(in_channels, out_channels, l, padding='same')

    def forward(self, x):

        return self.int_kern(x) + self.w(x)



class FNO2D(nn.Module):
    def __init__(self, modes_x, modes_y, width, l, n_layer=4, hidden_proj=None):
        super(FNO2D, self).__init__()

        """
        The overall network. It contains 3 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.P .
        2. 3 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.Q.

        input: the solution of the initial condition and location (x, bar(u(x)))
        input shape: (batchsize, x=s, c=2)
        output: the solution at a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes_x = modes_x  # This is k_modes
        self.modes_y = modes_y  # This is k_modes
        self.width = width  # This is d_v
        self.l = l #kernel size in linear transformation
        self.n_layer = n_layer #nbr of layers 
        
        if not hidden_proj:
            self.hidden_proj = self.width
        else:
            self.hidden_proj = hidden_proj

        self.padding = 0  # pad the domain if input is non-periodic

        self.activation = nn.LeakyReLU()

        # Lifting transformation, corresponding to a simple linear transformation
        self.P = nn.Linear(5, self.width)  # input channel is 5: (u(x,y),v(x,y),w,(x,y) x, y)
        
        
        # n_l sequential layers
        self.layers = nn.ModuleList()
        for i in range(self.n_layer):
            self.layers.append(FourierLayer2D(self.width, self.width, self.modes_x, self.modes_y, l))


        # Projection layer
        self.Q = nn.Sequential(nn.Linear(self.width, self.hidden_proj), self.activation, nn.Linear(self.hidden_proj, 1))
        #self.Q = nn.Sequential(nn.Linear(self.width, self.width), self.activation, nn.Linear(self.width, 1))


    def forward(self, x, input_grid):
        # input_grid must be the concatenation of X and Y (meshgrids) shape must be (x_size, y_size, 2)
        # shape of x is (batch_size, x_size, y_size)
        
        
        batch_size = x.shape[0]
        input_grid = input_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        # Concatenate the grid to the input to gt the term v0
        x = torch.cat((input_grid, x), dim=-1)

        # Apply the lifting transformation
        x = self.P(x)

        # Permute the axis, so that the axis corresponding to the physical space is the last (in order to compute the FFT)
        x = x.permute(0, 3, 1, 2)

        # Add padding (if input is non-periodic)
        if self.padding != 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])

        # Apply n_l integral layers
        
        for layer in self.layers:
            x = self.activation(layer(x))
  
        # Remove padding if input is non-periodic
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]

        # Comeback to the origin position of the axes
        # (batch_size, lifting_dimension, x_size, y_size) -> (batch_size, x_size, y_size, lifting_dimension)
        x = x.permute(0, 2, 3, 1)

        # Apply projection to go back to the original space (non-lifted)

        x = self.Q(x).squeeze(-1)
        return x