import torch
import torch.nn as nn
import torch.nn.functional as F


class IntegralKernel1D(nn.Module):
    """
    Linear transformation in Fourier SPace (part of 1D Fourier layer). It does FFT, linear transform, and Inverse FFT.
    """

    def __init__(self, in_channels, out_channels, modes1):
        super(IntegralKernel1D, self).__init__()

        self.in_channels = in_channels  # This is d_v
        self.out_channels = out_channels  # This is d_v
        self.modes1 = modes1  # This is k_max

        self.scale = (1 / (in_channels * out_channels))

        # Parametrization R of the kernel
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]  # Batch size
        
        n = x.shape[-1]  # Number of grid points where input and intermediate states are evaluated

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)
        # Use compl_mul1d to perform the multiplication between the relevant Fourier Modes and self.weights and fill the tensor out_tf with the corresponding values
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:,:,:self.modes1], self.weights)
        # Return to physical space
        x = torch.fft.irfft(out_ft, n=n)
        return x
    



class Fourier_layer1D(nn.Module):
    """ 
    Full Fourier Layer adding the Integral Kernel (linear operation in frequency space) and the linear operation (in real space).
    """

    def __init__(self, in_channels, out_channels, f_modes, l):
        super(Fourier_layer1D, self).__init__()

        self.integral_kernel = IntegralKernel1D(in_channels, out_channels, f_modes)
        self.linear_transformation = nn.Conv1d(in_channels, out_channels, l, padding="same")

    def forward(self, x):
        output = self.integral_kernel(x) + self.linear_transformation(x)
        return output
    



class FNO1D(nn.Module):
    def __init__(self, modes, width, l, n_layer=4, hidden_proj=None):
        super(FNO1D, self).__init__()

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

        self.modes = modes # This is k_modes
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
        self.P = nn.Linear(1, self.width)  # input channel is 5: (u(x,y),v(x,y),w,(x,y) x, y)
        
        
        # n_l sequential layers
        self.layers = nn.ModuleList()
        for i in range(self.n_layer):
            self.layers.append(Fourier_layer1D(self.width, self.width, self.modes, l))


        # Projection layer
        self.Q = nn.Sequential(nn.Linear(self.width, self.hidden_proj), self.activation, nn.Linear(self.hidden_proj, 1))
        #self.Q = nn.Sequential(nn.Linear(self.width, self.width), self.activation, nn.Linear(self.width, 1))


    def forward(self, x):
        # input_grid must be the concatenation of X and Y (meshgrids) shape must be (x_size, y_size, 2)
        # shape of x is (batch_size, x_size)
        
        batch_size = x.shape[0]

        # Apply the lifting transformation
        x.unsqueeze_(-1)  # (batch_size, x_size) -> (batch_size, x_size, 1)
        x = self.P(x)

        # Permute the axis, so that the axis corresponding to the physical space is the last (in order to compute the FFT)
        x = x.permute(0, 2, 1)  # (batch_size, x_size, lifting_dimension) -> (batch_size, lifting_dimension, x_size)

        # Add padding (if input is non-periodic)
        if self.padding != 0:
            x = F.pad(x, [0, self.padding])  # pad the last dimension (physical space)

        # Apply n_l integral layers
        
        for layer in self.layers:
            x = self.activation(layer(x))
  
        # Remove padding if input is non-periodic
        if self.padding != 0:
            x = x[..., :-self.padding]

        # Comeback to the origin position of the axes
        # (batch_size, lifting_dimension, x_size, y_size) -> (batch_size, x_size, y_size, lifting_dimension)
        x = x.permute(0, 2, 1)

        # Apply projection to go back to the original space (non-lifted)

        x = self.Q(x).squeeze(-1)
        return x
        