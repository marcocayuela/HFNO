import torch 
import numpy as np

import os 
import glob
import json
import csv
import h5py

import models.trainer as trainer
import models.hfno_2D as fno
from torch.utils.data import DataLoader, TensorDataset #to manage datasets and bash 

### FUNCTION DEFINITION ###

def print_carac(dt, ndim, nf, nk, re, resolution, time):

    print("dt: {} ; ndim: {} ; nf: {} ; nk: {} ; re: {} ; resolution: {} ; simulation time: {}".format(dt, ndim, nf, nk, re, resolution, time[-1]))

##### IMPORT DATA #####

print("Data importation...")

filename = "../data/kolmogorov/RE34/results9.h5"

with h5py.File(filename, "r") as f:

    dissipation = f["dissipation"][()]
    dt = f["dt"][()]
    ndim = f["ndim"][()]
    nf = f["nf"][()]
    nk = f["nk"][()]
    re = f["re"][()]
    resolution = f["resolution"][()]
    time = f["time"][()]
    velocity_field = f["velocity_field"][()]
    print_carac(dt, ndim, nf, nk, re, resolution, time)

print("Importation done!")

##### DATASET PREPROCESSING #####

print("Dataset pre-processing...")

p_val = 0.3
batch_size = 128

n = velocity_field.shape[0]
n_training = int(np.floor((1-p_val)*n))

velocity_field = torch.from_numpy(velocity_field).type(dtype=torch.float32)
input_train = velocity_field[:n_training,...]
output_train = velocity_field[1:(n_training+1)]
input_test = velocity_field[n_training:-1]
output_test = velocity_field[n_training+1:]


n_train = input_train.shape[0]
n_test = input_test.shape[0]

training_set = DataLoader(TensorDataset(input_train, output_train),
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0,
                          persistent_workers=False,
                          pin_memory=False)

testing_set = DataLoader(TensorDataset(input_test, output_test),
                         batch_size=n_test,
                         shuffle=False,
                         num_workers=0,
                         persistent_workers=False,
                         pin_memory=False)


print("Pre-processing done!")

### MODEL ###



k_targets = {0.7: 0.007215375318230077,
 0.8: 0.036076876591150386,
 0.85: 0.036076876591150386,
 0.9: 0.05050762722761054,
 0.95: 0.07936912850053085,
 0.99: 0.10823062977345116,
 0.999: 0.15152288168283162,
 0.99999: 0.2236766348651324}

modes = [4,8]
#modes_start = [k_targets[0.9]*64]
#modes_end = [k_targets[0.95]*64, k_targets[0.999]*64]
depth = 64
width_MLP = 128
n_layers_MLP = 1
input_size = 2
output_size = 2

res = 64
#last_modes_cutoff = modes[1]

model = fno.HFNO2D_l2(modes, depth, width_MLP, n_layers_MLP, input_size, output_size, res)
#model = fno.FNO2D_v2_overlap(modes_start, modes_end, depth, width_MLP, n_layers_MLP, input_size, output_size, res)
#model = fno.FNO2D(modes, 8, depth, width_MLP, n_layers_MLP, input_size, output_size)
#model = fno.FNO2D_v2_CNN(modes, depth, width_MLP, n_layers_MLP, input_size, output_size, res, last_modes_cutoff=k_targets[0.99999]*64)

### TRAINING ###

print("Start training...")
epochs = 30
learning_rate = 0.001
wd = 0.001
loss_name = "MSE"

if loss_name=="MSE":
    loss = torch.nn.MSELoss()
elif loss_name=="L1":
    loss = torch.nn.L1Loss()


dict_loss = trainer.train_fourier(model=model,
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

    where = "fno"
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
                            "depth": depth,
                            "MLP_width": width_MLP,
                            "n_MLP": n_layers_MLP,
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
