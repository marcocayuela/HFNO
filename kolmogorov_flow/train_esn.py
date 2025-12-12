import csv
import json
import os 
import h5py

import torch
import numpy as np

from models.ESN import ESN


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




## ENTER MODEL ###
num_model = 1
##################

last_mode_ESN = 15
hidden_size = 10000
washout = [100]
alpha = 1e-4


model_name = "model_"+str(num_model)+".pt"
fno_model = torch.load(os.path.join("./trained_models/fno",model_name))

csv_filename = './trained_models/fno/model_hyperparameters.csv'
with open(csv_filename, mode='r', newline='', encoding='utf-8') as file:

    reader = csv.DictReader(file)
   
    for i, row in enumerate(reader, start=1):
        if row["model_name"] == "model_"+str(num_model):
            model_info = row
            break


losses_filename = "losses_"+str(num_model)+".json"
with open(os.path.join("./trained_models/fno",losses_filename), 'r', encoding='utf-8') as file:
    
    dict_losses = json.load(file)



p_val = 0.3

n = velocity_field.shape[0]
n_training = int(np.floor((1-p_val)*n))

if isinstance(velocity_field, np.ndarray):
    velocity_field = torch.from_numpy(velocity_field).type(dtype=torch.float32)
input_train = velocity_field[:n_training,...]
output_train = velocity_field[1:(n_training+1)]
input_test = velocity_field[n_training:-1]
output_test = velocity_field[n_training+1:]


n_train = input_train.shape[0]
n_test = input_test.shape[0]



def esn_input(input_velocity, k_cutoff_start, k_cutoff_end):

    kxx = np.fft.fftfreq(resolution, d=1.0) 
    kyy = np.fft.fftfreq(resolution, d=1.0)  
    kx, ky = np.meshgrid(kxx, kyy, indexing='ij')
    k_magnitude = np.sqrt(kx**2 + ky**2) 
    rk_magnitude = k_magnitude[:,:resolution//2+1]

    mask = np.logical_and(rk_magnitude <= k_cutoff_end, rk_magnitude >= k_cutoff_start)

    velocity_field = input_velocity
    fft_vel = torch.fft.rfft2(velocity_field, axis=(1, 2))
    filtered_fft_vel = fft_vel[:,mask,:]
    filtered_fft_vel = torch.concatenate((filtered_fft_vel.real.unsqueeze(-1),filtered_fft_vel.imag.unsqueeze(-1)), axis=-1)
    W_input = filtered_fft_vel.reshape(filtered_fft_vel.shape[0],-1)

    print("Input size: {}".format(W_input.shape[-1]))
    return W_input

def esn_output(residuals, k_cutoff_end, k_cutoff_start=0):

    kxx = np.fft.fftfreq(resolution, d=1.0) 
    kyy = np.fft.fftfreq(resolution, d=1.0)  
    kx, ky = np.meshgrid(kxx, kyy, indexing='ij')
    k_magnitude = np.sqrt(kx**2 + ky**2) 
    rk_magnitude = k_magnitude[:,:resolution//2+1]

    mask = np.logical_and(rk_magnitude <= k_cutoff_end, rk_magnitude >= k_cutoff_start)

    fft_res = torch.fft.rfft2(residuals, axis=(1,2))
    filtered_fft_res = fft_res[:,mask,:]
    filtered_fft_res = torch.concatenate((filtered_fft_res.real.unsqueeze(-1),filtered_fft_res.imag.unsqueeze(-1)), axis=-1)
    W_output = filtered_fft_res.reshape(filtered_fft_res.shape[0],-1)

    print("Output size: {}".format(W_output.shape[-1]))
    return W_output



k_start_W = fno_model.modes[-1]/resolution
k_end_W = last_mode_ESN/resolution

with torch.no_grad():
    preds_fno_train = fno_model(input_train)
    preds_fno_test = fno_model(input_test)

    residuals_train = output_train - preds_fno_train
    residuals_test = output_test - preds_fno_test


W_input_train = esn_input(input_train, k_start_W, k_end_W)
W_output_train = esn_output(residuals_train, k_end_W, k_start_W)

W_input_test = esn_input(input_test, k_start_W, k_end_W)
W_output_test = esn_output(residuals_test, k_end_W, k_start_W)
k_end_W




print("Start training...")

input_size = W_input_train.shape[-1]

output_size = W_output_train.shape[-1]

esn = ESN(input_size=input_size, hidden_size=hidden_size, output_size=output_size, alpha=alpha)

# 3. Entraînement
esn.train(W_input_train, W_output_train, washout=100)

# 4. Prédiction
predictions = esn.predict(W_input_train)  # [2300, 128]

print("Training error:", torch.mean(torch.sqrt(torch.mean((predictions.squeeze() - W_output_train[washout[0]:])**2,dim=1)/torch.mean(W_output_train[washout[0]:]**2, dim=1))).item())


print("Training done!")


### results of the model ###
with torch.no_grad():
    output_pred_batch = esn.predict(W_input_train)
    output_batch = W_output_train[washout[0]:]
    rel_train_err = torch.mean(torch.sqrt(torch.mean((output_pred_batch-output_batch)**2, dim=1)/torch.mean(output_batch**2, dim=1))).item()
    


    output_pred_batch = esn.predict(W_input_test)
    output_batch = W_output_test[washout[0]:]
    rel_test_err = torch.mean(torch.sqrt(torch.mean((output_pred_batch-output_batch)**2, dim=1)/torch.mean(output_batch**2, dim=1))).item()


print("In Fourier space\nTrainig error: {:.3f}\nTesting error: {:.3f}\n".format(rel_train_err, rel_test_err))


output_esn_train = np.zeros((n_train-washout[0], resolution, resolution//2+1, 2), dtype=np.cfloat)
output_esn_test = np.zeros((n_test-washout[0], resolution, resolution//2+1, 2), dtype=np.cfloat)

pred_esn_train = esn.predict(W_input_train)
pred_esn_test = esn.predict(W_input_test)
pred_esn_train_rs = pred_esn_train.reshape(pred_esn_train.shape[0],pred_esn_train.shape[-1]//4,2,2)
pred_esn_train_modes = pred_esn_train_rs[...,0] + 1j*pred_esn_train_rs[...,1]

pred_esn_test_rs = pred_esn_test.reshape(pred_esn_test.shape[0],pred_esn_train.shape[-1]//4,2,2)
pred_esn_test_modes = pred_esn_test_rs[...,0] + 1j*pred_esn_test_rs[...,1]


kxx = np.fft.fftfreq(resolution, d=1.0) 
kyy = np.fft.fftfreq(resolution, d=1.0)  
kx, ky = np.meshgrid(kxx, kyy, indexing='ij')
k_magnitude = np.sqrt(kx**2 + ky**2) 
rk_magnitude = k_magnitude[:,:resolution//2+1]

mask = np.logical_and(rk_magnitude <= k_end_W, rk_magnitude >= k_start_W)
#mask = rk_magnitude <= k_end_W

output_esn_train[:,mask,:] = pred_esn_train_modes
output_esn_test[:,mask,:] = pred_esn_test_modes

output_esn_ps_train = np.fft.irfft2(output_esn_train, axes=(1,2))
output_esn_ps_test = np.fft.irfft2(output_esn_test, axes=(1,2))
full_pred_train = preds_fno_train[washout[0]:] + output_esn_ps_train
full_pred_test = preds_fno_test[washout[0]:] + output_esn_ps_test
fno_error_train = torch.mean(torch.sqrt(torch.mean((preds_fno_train-output_train)**2, axis=(1,2))/torch.mean(output_train**2, axis=(1,2))))
full_error_train = torch.mean(torch.sqrt(torch.mean((full_pred_train-output_train[washout[0]:])**2, axis=(1,2))/torch.mean(output_train[washout[0]:]**2, axis=(1,2))))
fno_error_test = torch.mean(torch.sqrt(torch.mean((preds_fno_test-output_test)**2, axis=(1,2))/torch.mean(output_test**2, axis=(1,2))))
full_error_test = torch.mean(torch.sqrt(torch.mean((full_pred_test-output_test[washout[0]:])**2, axis=(1,2))/torch.mean(output_test[washout[0]:]**2, axis=(1,2))))


print("In physical space\nError fno train: {:.3f}\nError fno test: {:.3f}\nError full model train: {:.3f}\nError full model test: {:.3f}".format(fno_error_train, fno_error_test, full_error_train, full_error_test))

esn_pred_train = torch.mean(torch.sqrt(torch.mean((torch.tensor(output_esn_ps_train)-residuals_train[washout[0]:])**2, axis=(1,2))/torch.mean(residuals_train[washout[0]:]**2, axis=(1,2))))
esn_pred_test = torch.mean(torch.sqrt(torch.mean((torch.tensor(output_esn_ps_test)-residuals_test[washout[0]:])**2, axis=(1,2))/torch.mean(residuals_test[washout[0]:]**2, axis=(1,2))))
print("In physical space\nError esn train: {:.3f}\nError esn test: {:.3f}".format(esn_pred_train, esn_pred_test))


# Sauvegarde du modèle ESN
combined_model_dict = {
    "fno_model": fno_model,
    "esn_model": esn,
    "model_hyperparameters": model_info,
    "losses": dict_losses,
    "fno_error_train": fno_error_train.item(),
    "fno_error_test": fno_error_test.item(),
    "full_error_train": full_error_train.item(),
    "full_error_test": full_error_test.item(),
    "esn_pred_train": esn_pred_train.item(),
    "esn_pred_test": esn_pred_test.item()
}

model_name = input("Do you want to save the model ?: type no if not, else the name of the model: ")

if model_name != "no":
    save_path = os.path.join("./trained_models_esn",model_name + ".pt")
    torch.save(combined_model_dict, save_path)

    print(f"Combined model dictionary saved at {save_path}")