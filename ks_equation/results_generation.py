import torch 
import numpy as np

import os 
import glob
import json
import csv
import sys

import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
main_dir = os.path.dirname(current_file)    
project_root = os.path.abspath(os.path.join(main_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.trainer import *
from models.hfno_1D import *

from torch.utils.data import DataLoader, TensorDataset #to manage datasets and bash 




def grid_encoding(functions, x_coordinates, encoding_dimension):
    """ 
    This function reduces the size of a the input tensor functions and the corresponding x_coordinates to a lower size (encodinf_dimension) according the x dimension

    :param functions: this is a tensor of dimension (P, M) representing a 1D function u evolving through time, each coordinate (i,j) corresponds to a location x_i and a time t_j i.e function[i,j] = u(x_i, t_j)
    :param x_coordinates: this is a tensor of dimension (P,) representing the points (x_i) where the function u has been evaluated.
    :param encoding_dimension: this is an integer descibing the new size of the x axis 

    :return encoding_coordinates: tensor of dimension (encoding_dimension,) representing the x points where the functions is now evaluated
    :return encoded_functions: tensor of dimension (encoding_dimension, M) representing the functions on a lower space.
    """
    P = functions.shape[0]
    M = functions.shape[1]
    
    idx = np.linspace(0, P-1, encoding_dimension)
    encoded_functions = functions[idx, :]
    encoding_coordinates = x_coordinates[idx]
    
    return encoding_coordinates, encoded_functions





##########################
m = 256
nu = 0.7
model_name = "model_1"

##########################

data = torch.from_numpy(np.load(os.path.join(os.path.abspath(''), "../data/ks_equation/data_"+str(int(100*nu))+".npy"))).type(torch.float32)
mat_deriv = torch.from_numpy(np.load(os.path.join(os.path.abspath(''), "../data/ks_equation/mat_deriv_"+str(int(100*nu))+".npy"))).type(torch.float32)
grid_carac = torch.from_numpy(np.load(os.path.join(os.path.abspath(''), "../data/ks_equation/grid_"+str(int(100*nu))+".npy"))).type(torch.float32)
nt, dt, Tf, Mx, dx, L = grid_carac
u_test = data[:,100:]
mat_deriv_test = mat_deriv[:,100:]

x_coordinates = torch.arange(0., Mx)*dx # reconstruction of the vector encoding the coordinates used for x

n_test = u_test.shape[1]

x_encoded, encoded_input_test = grid_encoding(u_test, x_coordinates, encoding_dimension=m)
encoded_input_test = encoded_input_test.T

_, encoded_output_test = grid_encoding(mat_deriv_test, x_coordinates, encoding_dimension=m)
encoded_output_test = encoded_output_test.T


data = torch.from_numpy(np.load(os.path.join(os.path.abspath(''), "../data/ks_equation/data_"+str(int(100*nu))+"_test.npy"))).type(torch.float32)
mat_deriv = torch.from_numpy(np.load(os.path.join(os.path.abspath(''), "../data/ks_equation/mat_deriv_"+str(int(100*nu))+"_test.npy"))).type(torch.float32)
grid_carac = torch.from_numpy(np.load(os.path.join(os.path.abspath(''), "../data/ks_equation/grid_"+str(int(100*nu))+"_test.npy"))).type(torch.float32)
nt, dt, Tf, Mx, dx, L = grid_carac
u_train = data[:,100:]
mat_deriv_train = mat_deriv[:,100:]

x_coordinates = torch.arange(0., Mx)*dx # reconstruction of the vector encoding the coordinates used for x

n_train = u_train.shape[1]

_, encoded_input_train = grid_encoding(u_train, x_coordinates, encoding_dimension=m)
encoded_input_train = encoded_input_train.T

_, encoded_output_train = grid_encoding(mat_deriv_train, x_coordinates, encoding_dimension=m)
encoded_output_train = encoded_output_train.T






model = torch.load(os.path.join(os.path.abspath(''), "trained_models/fno/"+model_name+".pt"), weights_only=False)
with torch.no_grad():
    preds_train = model(encoded_input_train)
    preds_train_1 = model(encoded_input_train, indexes=[0], sum_w=False)
    preds_train_2 = model(encoded_input_train, indexes=[1], sum_w=False)
    preds_train_w = model(encoded_input_train, indexes=0, sum_w=True)


error_train = torch.abs(preds_train-encoded_output_train)



L = 22  # Domain
T = 150  # Time

x = torch.linspace(0, L, encoded_output_train.shape[0])
t = torch.linspace(0, T, encoded_output_train.shape[1])


vmin = min(encoded_output_train.min(), preds_train.min())
vmax = max(encoded_output_train.max(), preds_train.max())

fig, axs = plt.subplots(6, 1, figsize=(8,8))

im1 = axs[0].imshow(encoded_output_train.T, extent=[0, L, 0, T], aspect='auto', origin='lower', cmap='bwr')
axs[0].set_title("Target", fontsize=10)
axs[0].set_ylabel(r"Physical Domain $(x)$", fontsize=8)
#axs[0].set_xlabel(r"Time $(t)$", fontsize=8)
fig.colorbar(im1, ax=axs[0], orientation='vertical', fraction=0.02, pad=0.04)

im2 = axs[1].imshow(preds_train.T, extent=[0, L, 0, T], aspect='auto', origin='lower', cmap='bwr')
axs[1].set_title("Prediction", fontsize=10)
axs[1].set_ylabel(r"Physical Domain $(x)$", fontsize=8)
#axs[1].set_xlabel(r"Time $(t)$", fontsize=8)
fig.colorbar(im2, ax=axs[1], orientation='vertical', fraction=0.02, pad=0.04)

im3 = axs[2].imshow(error_train.T, extent=[0, L, 0, T], aspect='auto', origin='lower', cmap='inferno', vmin=0, vmax=1)
axs[2].set_title(f"Error $|\hat u - u|$", fontsize=10)
axs[2].set_ylabel(r"Physical Domain $(x)$", fontsize=8)
axs[2].set_xlabel(r"Time $(t)$", fontsize=8)
fig.colorbar(im3, ax=axs[2], orientation='vertical', fraction=0.02, pad=0.04)

im4 = axs[3].imshow(preds_train_1.T, extent=[0, L, 0, T], aspect='auto', origin='lower', cmap='bwr')
axs[3].set_title(f"First Fourier Layer$", fontsize=10)
axs[3].set_ylabel(r"Physical Domain $(x)$", fontsize=8)
axs[3].set_xlabel(r"Time $(t)$", fontsize=8)
fig.colorbar(im4, ax=axs[3], orientation='vertical', fraction=0.02, pad=0.04)

im5 = axs[4].imshow(preds_train_2.T, extent=[0, L, 0, T], aspect='auto', origin='lower', cmap='bwr')
axs[4].set_title(f"Second Fourier Layer$", fontsize=10)
axs[4].set_ylabel(r"Physical Domain $(x)$", fontsize=8)
axs[4].set_xlabel(r"Time $(t)$", fontsize=8)
fig.colorbar(im5, ax=axs[4], orientation='vertical', fraction=0.02, pad=0.04)

im6 = axs[5].imshow(preds_train_w.T, extent=[0, L, 0, T], aspect='auto', origin='lower', cmap='bwr')
axs[5].set_title(f"Linear residual predictor$", fontsize=10)
axs[5].set_ylabel(r"Physical Domain $(x)$", fontsize=8)
axs[5].set_xlabel(r"Time $(t)$", fontsize=8)
fig.colorbar(im6, ax=axs[5], orientation='vertical', fraction=0.02, pad=0.04)



# Ajuster la mise en page
plt.tight_layout()
if not os.path.exists("plots"):
    os.mkdir("plots")
plt.savefig("plots/"+model_name+".png")
plt.show()
