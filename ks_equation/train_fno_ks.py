import torch 
import numpy as np

import os 
import glob
import json
import csv
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import models.trainer as trainer
import models.fno_1D as fno
from torch.utils.data import DataLoader, TensorDataset #to manage datasets and bash 

### FUNCTION DEFINITION ###

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


##### IMPORT DATA #####

m = 256
nu = 0.7


data = torch.from_numpy(np.load(os.path.join(os.path.abspath(''), "../data/ks_equation/data_"+str(int(100*nu))+".npy"))).type(torch.float32)
mat_deriv = torch.from_numpy(np.load(os.path.join(os.path.abspath(''), "../data/ks_equation/mat_deriv_"+str(int(100*nu))+".npy"))).type(torch.float32)
grid_carac = torch.from_numpy(np.load(os.path.join(os.path.abspath(''), "../data/ks_equation/grid_"+str(int(100*nu))+".npy"))).type(torch.float32)
nt, dt, Tf, Mx, dx, L = grid_carac
u = data[:,100:]
mat_deriv = mat_deriv[:,100:]

x_coordinates = torch.arange(0., Mx)*dx # reconstruction of the vector encoding the coordinates used for x

n_train = u.shape[1]

x_encoded, encoded_input = grid_encoding(u, x_coordinates, encoding_dimension=m)
encoded_input_train = encoded_input.T

_, encoded_output = grid_encoding(mat_deriv, x_coordinates, encoding_dimension=m)
encoded_output_train = encoded_output.T


data = torch.from_numpy(np.load(os.path.join(os.path.abspath(''), "../data/ks_equation/data_"+str(int(100*nu))+"_test.npy"))).type(torch.float32)
mat_deriv = torch.from_numpy(np.load(os.path.join(os.path.abspath(''), "../data/ks_equation/mat_deriv_"+str(int(100*nu))+"_test.npy"))).type(torch.float32)
grid_carac = torch.from_numpy(np.load(os.path.join(os.path.abspath(''), "../data/ks_equation/grid_"+str(int(100*nu))+"_test.npy"))).type(torch.float32)
nt, dt, Tf, Mx, dx, L = grid_carac
u = data[:,100:]
mat_deriv = mat_deriv[:,100:]

x_coordinates = torch.arange(0., Mx)*dx # reconstruction of the vector encoding the coordinates used for x

n_test = u.shape[1]

_, encoded_input = grid_encoding(u, x_coordinates, encoding_dimension=m)
encoded_input_test = encoded_input.T

_, encoded_output = grid_encoding(mat_deriv, x_coordinates, encoding_dimension=m)
encoded_output_test = encoded_output.T


##### DATASET PREPROCESSING #####

print("Dataset pre-processing...")

batch_size = 64

training_set = DataLoader(TensorDataset(encoded_input_test, encoded_output_test), batch_size=batch_size, shuffle=True)
testing_set = DataLoader(TensorDataset(encoded_input_train, encoded_output_train), batch_size=n_test, shuffle=False)

print(encoded_input_test.shape)
print("Pre-processing done!")

### MODEL ###

modes = 15

width = 64
n_layers = 1
l = 1
input_size = 1
output_size = 1


model = fno.FNO1D(modes, width, l, n_layers)



print("Start training...")

epochs = 100
learning_rate = 0.001
wd = 0.000001
loss_name = "MSE"

if loss_name=="MSE":
    loss = torch.nn.MSELoss()
elif loss_name=="L1":
    loss = torch.nn.L1Loss()


dict_loss = trainer.train_fourier_1D(model=model,
                                  training_set=training_set,
                                  testing_set=testing_set,
                                  epochs=epochs,
                                  learning_rate=learning_rate,
                                  validation_set=None,
                                  wd=wd,
                                  l=loss)

print("Training done!")

with torch.no_grad():

    model.eval()
    rel_train_err = 0.
    for _, (input_batch, output_batch) in enumerate(training_set):
        output_pred_batch = model(input_batch)
        rel_train_err += torch.sqrt(torch.mean((output_pred_batch-output_batch)**2)/torch.mean(output_batch**2)).item()
    rel_train_err /= len(training_set)

    rel_test_err = 0.
    for _, (input_batch, output_batch) in enumerate(testing_set):
        output_pred_batch = model(input_batch)
        rel_test_err += torch.sqrt(torch.mean((output_pred_batch-output_batch)**2)/torch.mean(output_batch**2)).item()
    rel_test_err /= len(testing_set)

    rel_val_err = 0

print("Trainig error: {:.3f}\nTesting error: {:.3f}".format(rel_train_err, rel_test_err))

### SAVING THE MODEL ###

save = input("Do you want to save the model ?: press \"y\" for yes, anything else for no: ")

if save == "y":

    where = "fno_classic"
    if not os.path.exists(os.path.join(".","trained_models",where)):
        os.makedirs(os.path.join(".","trained_models",where))

    index = 0
    for file in os.listdir(os.path.join(".","trained_models",where)):
        if file.startswith("model_") and file.endswith(".pt"):
            try:
                curr_index = int(file[len("model_"):-3])
                if curr_index > index:
                    index = curr_index
            except ValueError:
                pass

    torch.save(model, os.path.join(".","trained_models",where,"model_"+ str(index+1)+".pt"))
    with open(os.path.join(".","trained_models",where,"losses_"+str(index+1)+".json"), "w") as fichier:
        json.dump(dict_loss, fichier)

    print("The model has been saved with name {} in {}".format("model_"+ str(index+1)+".pt", os.path.join(".","trained_models",where)))


    model_hyperparameters = {"model_name": "model_"+ str(index+1),
                            "modes": "-".join(map(str, modes)),
                            "width": width,
                            "n_layers": n_layers,
                            "learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "epochs": epochs,
                            "w_pen": wd,
                            "loss_name": loss_name,
                            "final_test_error": rel_val_err}
                            

    csv_filename = "model_hyperparameters.csv"

    file_exists = os.path.exists(os.path.join(".","trained_models",where,csv_filename))

    with open(os.path.join(".","trained_models",where,csv_filename), mode='a', newline='') as file:
    
        writer = csv.DictWriter(file, fieldnames=model_hyperparameters.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(model_hyperparameters)

    print("The CSV file {} has been updated".format(os.path.join(".","trained_models",where,csv_filename)))
