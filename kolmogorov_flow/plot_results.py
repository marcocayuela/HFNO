import csv
import json
import os 
import h5py

import torch
import numpy as np

import sys

# Ajouter le chemin du dossier parent au sys.path
# Use the current working directory as the base path in Jupyter Notebook
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# Importer le fichier ESN.py depuis le sous-dossier
from models.ESN import ESN

import models.trainer as trainer
import models.hfno_2D_ww as fno
from torch.utils.data import DataLoader, TensorDataset #to manage datasets and bash 

import matplotlib.pyplot as plt



### DATA IMPORT
# Load the data

### FUNCTION DEFINITION ###

def print_carac(dt, ndim, nf, nk, re, resolution, time):

    print("dt: {} ; ndim: {} ; nf: {} ; nk: {} ; re: {} ; resolution: {} ; simulation time: {}".format(dt, ndim, nf, nk, re, resolution, time[-1]))

##### IMPORT DATA #####

print("Data importation...")

filename = "../data/kolmogorov/RE90/results2.h5"

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


last_mode_ESN = 15
washout=[100]
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



### MODEL LOADING ###


model_name = "re90_4_8_15"
loaded_combined_model_dict = torch.load(os.path.join("./trained_models_esn",model_name+".pt"))

# Accéder aux modèles
fno_model = loaded_combined_model_dict["fno_model"]
esn_model = loaded_combined_model_dict["esn_model"]

print("Combined model dictionary loaded successfully!")



k_start_esn = fno_model.modes[-1]/resolution
k_end_esn = last_mode_ESN/resolution



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


with torch.no_grad():
    preds_fno_train = fno_model(input_train)
    preds_fno_test = fno_model(input_test)

    residuals_train = output_train - preds_fno_train
    residuals_test = output_test - preds_fno_test


W_input_train = esn_input(input_train, k_start_esn, k_end_esn)
W_output_train = esn_output(residuals_train, k_end_esn, k_start_esn)

W_input_test = esn_input(input_test, k_start_esn, k_end_esn)
W_output_test = esn_output(residuals_test, k_end_esn, k_start_esn)




### Prediction with ESN ###

if len(fno_model.modes)==2:
    output_esn_train = np.zeros((n_train-washout[0], resolution, resolution//2+1, 2), dtype=np.cfloat)
    output_esn_test = np.zeros((n_test-washout[0], resolution, resolution//2+1, 2), dtype=np.cfloat)

    pred_esn_train = esn_model.predict(W_input_train)
    pred_esn_test = esn_model.predict(W_input_test)
    pred_esn_train_rs = pred_esn_train.reshape(pred_esn_train.shape[0],pred_esn_train.shape[-1]//2,2)
    pred_esn_train_modes = pred_esn_train_rs[...,0] + 1j*pred_esn_train_rs[...,1]

    pred_esn_test_rs = pred_esn_test.reshape(pred_esn_test.shape[0],pred_esn_train.shape[-1]//2,2)
    pred_esn_test_modes = pred_esn_test_rs[...,0] + 1j*pred_esn_test_rs[...,1]


    kxx = np.fft.fftfreq(resolution, d=1.0) 
    kyy = np.fft.fftfreq(resolution, d=1.0)  
    kx, ky = np.meshgrid(kxx, kyy, indexing='ij')
    k_magnitude = np.sqrt(kx**2 + ky**2) 
    rk_magnitude = k_magnitude[:,:resolution//2+1]

    mask = np.logical_and(rk_magnitude <= k_end_esn, rk_magnitude >= k_start_esn)
    output_esn_train = np.zeros((n_train-washout[0], resolution, resolution//2+1, 2), dtype=np.cfloat)
    output_esn_test = np.zeros((n_test-washout[0], resolution, resolution//2+1, 2), dtype=np.cfloat)

    pred_esn_train = esn_model.predict(W_input_train)
    pred_esn_test = esn_model.predict(W_input_test)
    pred_esn_train_rs = pred_esn_train.reshape(pred_esn_train.shape[0],pred_esn_train.shape[-1]//4,2,2)
    pred_esn_train_modes = pred_esn_train_rs[...,0] + 1j*pred_esn_train_rs[...,1]

    pred_esn_test_rs = pred_esn_test.reshape(pred_esn_test.shape[0],pred_esn_train.shape[-1]//4,2,2)
    pred_esn_test_modes = pred_esn_test_rs[...,0] + 1j*pred_esn_test_rs[...,1]


    kxx = np.fft.fftfreq(resolution, d=1.0) 
    kyy = np.fft.fftfreq(resolution, d=1.0)  
    kx, ky = np.meshgrid(kxx, kyy, indexing='ij')
    k_magnitude = np.sqrt(kx**2 + ky**2) 
    rk_magnitude = k_magnitude[:,:resolution//2+1]

    mask = np.logical_and(rk_magnitude <= k_end_esn, rk_magnitude >= k_start_esn)
    #mask = rk_magnitude <= k_end_W

    output_esn_train[:,mask,:] = pred_esn_train_modes
    output_esn_test[:,mask,:] = pred_esn_test_modes

    output_esn_ps_train = np.fft.irfft2(output_esn_train, axes=(1,2))
    output_esn_ps_test = np.fft.irfft2(output_esn_test, axes=(1,2))
    full_pred_train = preds_fno_train[100:] + output_esn_ps_train
    full_pred_test = preds_fno_test[100:] + output_esn_ps_test


    preds_fno_1_train = fno_model(input_train, indexes=[0])
    preds_fno_2_train = fno_model(input_train, indexes=[1])
    preds_fno_1_test = fno_model(input_test, indexes=[0])
    preds_fno_2_test = fno_model(input_test, indexes=[1])


    error_train = output_train[100:] - full_pred_train
    error_test = output_test[100:] - full_pred_test


    save=False
    if not os.path.exists(os.path.join("./plot_models",model_name)):
        os.mkdir(os.path.join("./plot_models",model_name))
        save_path = os.path.join("./plot_models", model_name)
        save = True


    ### PLOT RESULTS ###
    def compute_classical_energy_spectrum(u):
        """
        Spectre classique par FFT directe des composantes de vitesse
        u : ndarray (t, nx, ny, 2)
        Returns: k_vals_classical, E_k_classical
        """
        tsteps, nx, ny, _ = u.shape
        E_k_sum = 0.0

        for t in range(tsteps):
            ux = u[t, :, :, 0]
            uy = u[t, :, :, 1]

            uxf = np.fft.fft2(ux)
            uyf = np.fft.fft2(uy)

            E2D = 0.5 * (np.abs(uxf)**2 + np.abs(uyf)**2) / (nx * ny)
            E2D = np.fft.fftshift(E2D)

            # Build radial wavenumber grid
            kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx
            ky = np.fft.fftshift(np.fft.fftfreq(ny)) * ny
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            k_mag = np.sqrt(KX**2 + KY**2)

            k_max = int(np.max(k_mag))
            E_k = np.zeros(k_max)
            for i in range(k_max):
                mask = (k_mag >= i) & (k_mag < i + 1)
                E_k[i] += np.sum(E2D[mask])

            E_k_sum += E_k

        E_k_classical = E_k_sum / tsteps
        k_vals_classical = np.arange(len(E_k_classical))
        return k_vals_classical, E_k_classical
        

    k_bin_centers, spectrum_binned_data = compute_classical_energy_spectrum(velocity_field)
    _, spectrum_binned_fno_1 = compute_classical_energy_spectrum(preds_fno_1_train.detach().numpy())
    _, spectrum_binned_fno_2 = compute_classical_energy_spectrum(preds_fno_2_train.detach().numpy())
    _, spectrum_binned_esn = compute_classical_energy_spectrum(output_esn_ps_train)
    _, spectrum_binned_full = compute_classical_energy_spectrum(full_pred_train.detach().numpy())
    #_, spectrum_binned_error = compute_classical_energy_spectrum(error_train.detach().numpy())

    cumulative_energy = np.cumsum(spectrum_binned_data)
    total_energy = cumulative_energy[-1]
    normalized_cumulative_energy = cumulative_energy / total_energy

    # Find the wavenumber where cumulative energy reaches 80%
    plt.figure(figsize=(10, 6))
    plt.loglog(k_bin_centers, spectrum_binned_fno_1, marker='o', linestyle='-', color='b', label='Energy Spectrum of FNO 1st lay. output')
    plt.loglog(k_bin_centers, spectrum_binned_fno_2, marker='o', linestyle='-', color='purple', label='Energy Spectrum of FNO 2nd lay. output')
    plt.loglog(k_bin_centers, spectrum_binned_esn, marker='o', linestyle='-', color='g', label='Energy Spectrum of ESN output')
    plt.loglog(k_bin_centers, spectrum_binned_full, marker='o', linestyle='-', color='black', label='Energy Spectrum of full output')
    #plt.loglog(k_bin_centers, spectrum_binned_error, marker='o', linestyle='-', color='pink', label='Energy Spectrum of the error')
    plt.axvline(fno_model.modes[0], color='blue', linestyle='--', label=f'{np.round(normalized_cumulative_energy[fno_model.modes[0]-1]*100,2)}% Energy')
    plt.axvline(fno_model.modes[1], color='green', linestyle='--', label=f'{np.round(normalized_cumulative_energy[fno_model.modes[1]-1]*100,4)}% Energy')
    plt.axvline(last_mode_ESN, color='purple', linestyle='--', label=f'{np.round(normalized_cumulative_energy[last_mode_ESN-1]*100,7)}% Energy')

    plt.loglog(k_bin_centers, spectrum_binned_data, marker='o', linestyle='-', color='r', label='Energy Spectrum of raw data')


    plt.ylim(1e-13,1e4)
    plt.xlabel('Wavenumber |k| (log scale)')
    plt.ylabel('Energy Spectrum E(k) (log scale)')
    plt.title('Energy Spectrum in Frequency Domain (Log-Log Plot)')
    plt.grid(True)
    plt.legend()
    plt.show()

    if save:
        plt.savefig(os.path.join(save_path, "energy_spectrum_train.png"))


    def compute_vorticity(vel_field):
        u = vel_field[..., 0]  # shape (1000, 64, 64)
        v = vel_field[..., 1]  # shape (1000, 64, 64)

        dv_dx = torch.gradient(v, dim=2)[0]
        du_dy = torch.gradient(u, dim=1)[0] 
        vorticity = dv_dx - du_dy
        return vorticity


    fig, axs = plt.subplots(6,4, figsize=(15,12))

    snapshots_index = [0, 500, 1000, 1500]

    vort_output = compute_vorticity(output_train[100:])
    vort_fno_1 = compute_vorticity(preds_fno_1_train[100:])
    vort_fno_2 = compute_vorticity(preds_fno_2_train[100:])
    vort_esn = compute_vorticity(torch.tensor(output_esn_ps_train))
    vort_pred_train = compute_vorticity(full_pred_train)
    error = vort_output - vort_pred_train

    vmin = torch.min(torch.tensor([torch.min(vort_fno_1), torch.min(vort_fno_2), torch.min(vort_esn), torch.min(vort_output), torch.min(vort_pred_train), torch.min(error)]))
    vmax = torch.max(torch.tensor([torch.max(vort_fno_1), torch.max(vort_fno_2), torch.max(vort_esn),torch.max(vort_output), torch.max(vort_pred_train), torch.max(error)]))

    for i in range(4):
        axs[0,i].imshow(vort_output[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[1,i].imshow(vort_pred_train[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[3,i].imshow(vort_fno_1[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[4,i].imshow(vort_fno_2[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[5,i].imshow(vort_esn[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        c = axs[2,i].imshow(error[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[0,i].set_title("t = {}".format(snapshots_index[i]*0.1))

    for i in range(5):
        for j in range(4):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    for i in range(6):    
        row_titles = ["ground truth", "full pred", "error", "FN0 1st lay", "FN0 2nd lay", "ESN"]
        axs[i,0].set_ylabel(row_titles[i])


    plt.colorbar(c, ax=axs)
    fig.suptitle("Vorticity of differents snapshots in training set")
    plt.show()

    if save:
        plt.savefig(os.path.join(save_path, "vorticity_train.png"))



    ###Test###

    k_bin_centers, spectrum_binned_data = compute_classical_energy_spectrum(velocity_field)
    _, spectrum_binned_fno_1 = compute_classical_energy_spectrum(preds_fno_1_test.detach().numpy())
    _, spectrum_binned_fno_2 = compute_classical_energy_spectrum(preds_fno_2_test.detach().numpy())
    _, spectrum_binned_esn = compute_classical_energy_spectrum(output_esn_ps_test)
    _, spectrum_binned_full = compute_classical_energy_spectrum(full_pred_test.detach().numpy())
    _, spectrum_binned_error = compute_classical_energy_spectrum(error_test.detach().numpy())

    cumulative_energy = np.cumsum(spectrum_binned_data)
    total_energy = cumulative_energy[-1]
    normalized_cumulative_energy = cumulative_energy / total_energy

    # Find the wavenumber where cumulative energy reaches 80%
    plt.figure(figsize=(10, 6))
    plt.loglog(k_bin_centers, spectrum_binned_fno_1, marker='o', linestyle='-', color='b', label='Energy Spectrum of FNO 1st lay. output')
    plt.loglog(k_bin_centers, spectrum_binned_fno_2, marker='o', linestyle='-', color='purple', label='Energy Spectrum of FNO 2nd lay. output')
    plt.loglog(k_bin_centers, spectrum_binned_esn, marker='o', linestyle='-', color='g', label='Energy Spectrum of ESN output')
    plt.loglog(k_bin_centers, spectrum_binned_full, marker='o', linestyle='-', color='black', label='Energy Spectrum of full output')
    plt.loglog(k_bin_centers, spectrum_binned_error, marker='o', linestyle='-', color='pink', label='Energy Spectrum of the error')
    plt.axvline(fno_model.modes[0], color='blue', linestyle='--', label=f'{np.round(normalized_cumulative_energy[fno_model.modes[0]-1]*100,2)}% Energy')
    plt.axvline(fno_model.modes[1], color='green', linestyle='--', label=f'{np.round(normalized_cumulative_energy[fno_model.modes[1]-1]*100,4)}% Energy')
    plt.axvline(last_mode_ESN, color='purple', linestyle='--', label=f'{np.round(normalized_cumulative_energy[last_mode_ESN-1]*100,7)}% Energy')

    plt.loglog(k_bin_centers, spectrum_binned_data, marker='o', linestyle='-', color='r', label='Energy Spectrum of raw data')


    plt.ylim(1e-13,1e4)
    plt.xlabel('Wavenumber |k| (log scale)')
    plt.ylabel('Energy Spectrum E(k) (log scale)')
    plt.title('Energy Spectrum in Frequency Domain (Log-Log Plot)')
    plt.grid(True)
    plt.legend()
    plt.show()

    if save:
        plt.savefig(os.path.join(save_path, "energy_spectrum_test.png"))


    def compute_vorticity(vel_field):
        u = vel_field[..., 0]  # shape (1000, 64, 64)
        v = vel_field[..., 1]  # shape (1000, 64, 64)

        dv_dx = torch.gradient(v, dim=2)[0]
        du_dy = torch.gradient(u, dim=1)[0] 
        vorticity = dv_dx - du_dy
        return vorticity


    fig, axs = plt.subplots(6,4, figsize=(15,12))

    snapshots_index = [0, 500, 1000, 1500]

    vort_output = compute_vorticity(output_test[100:])
    vort_fno_1 = compute_vorticity(preds_fno_1_test[100:])
    vort_fno_2 = compute_vorticity(preds_fno_2_test[100:])
    vort_esn = compute_vorticity(torch.tensor(output_esn_ps_test))
    vort_pred_test = compute_vorticity(full_pred_test)
    error = vort_output - vort_pred_test

    vmin = torch.min(torch.tensor([torch.min(vort_fno_1), torch.min(vort_fno_2), torch.min(vort_esn), torch.min(vort_output), torch.min(vort_pred_test), torch.min(error)]))
    vmax = torch.max(torch.tensor([torch.max(vort_fno_1), torch.max(vort_fno_2), torch.max(vort_esn),torch.max(vort_output), torch.max(vort_pred_test), torch.max(error)]))

    for i in range(4):
        axs[0,i].imshow(vort_output[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[1,i].imshow(vort_pred_test[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[3,i].imshow(vort_fno_1[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[4,i].imshow(vort_fno_2[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[5,i].imshow(vort_esn[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        c = axs[2,i].imshow(error[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[0,i].set_title("t = {}".format(snapshots_index[i]*0.1))

    for i in range(5):
        for j in range(4):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    for i in range(6):    
        row_titles = ["ground truth", "full pred", "error", "FN0 1st lay", "FN0 2nd lay", "ESN"]
        axs[i,0].set_ylabel(row_titles[i])


    plt.colorbar(c, ax=axs)
    fig.suptitle("Vorticity of differents snapshots in testing set")
    plt.show()


    if save:
        plt.savefig(os.path.join(save_path, "vorticity_test.png"))


    ### WASSERSTEIN DISTANCE ###

    import ot 

    def wass_dist(vel0,vel1):

        X = np.sqrt(vel0[...,0]**2 + vel0[...,1]**2)
        Y = np.sqrt(vel1[...,0]**2 + vel1[...,1]**2)

        a = np.ones(len(X)) / len(X)
        b = np.ones(len(Y)) / len(Y)

        M = ot.dist(X, Y)
        M /= M.max()  # optional normalization

        # Compute Wasserstein distance (order 1)
        d = ot.emd2(a, b, M) 

        return d


    wass_dist_train = np.zeros((n_train-washout[0],))
    wass_dist_test = np.zeros((n_test-washout[0],))[::10]

    print(full_pred_test.shape, output_test.shape, full_pred_train.dtype, full_pred_test.dtype, output_train.dtype, output_test.dtype)
    print("Compute wasserstein distance\n")
    print("Train...\n")
    for i in range(n_train-washout[0]):
        wass_dist_train[i] = wass_dist(full_pred_train[i].detach().numpy(), output_train[100:][i].detach().numpy())
    #print("Test...\n")
    for i in range(wass_dist_test.shape[0]):    
        wass_dist_test[i] = wass_dist(full_pred_test[10*i].detach().numpy(), output_test[100:][10*i].detach().numpy())


    print(wass_dist_train.shape, wass_dist_test.shape, wass_dist_train[0], wass_dist_test[0])
    # Downsample the data for plotting
    wass_dist_train_downsampled = wass_dist_train[::10]  # Take every 10th point
    wass_dist_test_downsampled = wass_dist_test


    # Calcul de l'erreur L2 pour le training set
    l2_error_train = np.zeros((n_train - washout[0],))
    for i in range(n_train - washout[0]):
        l2_error_train[i] = np.sqrt(np.mean((full_pred_train[i].detach().numpy() - output_train[100:][i].detach().numpy())**2))

    # Downsample les données pour le plot
    l2_error_train_downsampled = l2_error_train[::10]
    wass_dist_train_downsampled = wass_dist_train[::10]

    # Plot Wasserstein distance et erreur L2 pour le training set
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Courbe de la distance de Wasserstein
    ax1.plot(np.arange(len(wass_dist_train_downsampled)) * dt * 10, wass_dist_train_downsampled, label="Wasserstein Distance (Train)", color="blue")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Wasserstein Distance", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True)

    # Ajouter un deuxième axe pour l'erreur L2
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(l2_error_train_downsampled)) * dt * 10, l2_error_train_downsampled, label="L2 Error (Train)", color="red")
    ax2.set_ylabel("L2 Error", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    # Ajouter une légende combinée
    fig.legend(loc="upper right", bbox_to_anchor=(1, 0.9), bbox_transform=ax1.transAxes)

    plt.title("Wasserstein Distance and L2 Error Over Time (Train)")
    plt.show()

    if save:
        plt.savefig(os.path.join(save_path, "error_train.png"))

    # Calcul de l'erreur L2 pour le test set
    l2_error_test = np.zeros((n_test - washout[0],))
    for i in range(n_test - washout[0]):
        l2_error_test[i] = np.sqrt(np.mean((full_pred_test[i].detach().numpy() - output_test[100:][i].detach().numpy())**2))

    # Downsample les données pour le plot
    l2_error_test_downsampled = l2_error_test[::10]
    #wass_dist_test_downsampled = wass_dist_test[::10]

    # Plot Wasserstein distance et erreur L2 pour le test set
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Courbe de la distance de Wasserstein
    ax1.plot(np.arange(len(wass_dist_test_downsampled)) * dt * 10, wass_dist_test_downsampled, label="Wasserstein Distance (Test)", color="blue")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Wasserstein Distance", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True)

    # Ajouter un deuxième axe pour l'erreur L2
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(l2_error_test_downsampled)) * dt * 10, l2_error_test_downsampled, label="L2 Error (Test)", color="red")
    ax2.set_ylabel("L2 Error", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    # Ajouter une légende combinée
    fig.legend(loc="upper right", bbox_to_anchor=(1, 0.9), bbox_transform=ax1.transAxes)

    plt.title("Wasserstein Distance and L2 Error Over Time (Test)")
    plt.show()

    if save:
        plt.savefig(os.path.join(save_path, "error_test.png"))



    fig, axs = plt.subplots(6,4, figsize=(15,12))

    snapshots_index = [0, 500, 1000, 1500]

    u_output = output_train[100:][...,0]
    u_fno_1 = preds_fno_1_train[100:][...,0]
    u_fno_2 = preds_fno_2_train[100:][...,0]
    u_esn = torch.tensor(output_esn_ps_train)[...,0]
    u_pred_train = full_pred_train[...,0]
    error = u_output - u_pred_train

    vmin = torch.min(torch.tensor([torch.min(u_fno_1), torch.min(u_fno_2),torch.min(u_esn), torch.min(u_output), torch.min(u_pred_train), torch.min(error)]))
    vmax = torch.max(torch.tensor([torch.max(u_fno_1), torch.max(u_fno_2),torch.max(u_esn),torch.max(u_output), torch.max(u_pred_train), torch.max(error)]))

    for i in range(4):
        axs[0,i].imshow(u_output[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[1,i].imshow(u_pred_train[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[3,i].imshow(u_fno_1[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[4,i].imshow(u_fno_2[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[5,i].imshow(u_esn[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        c = axs[2,i].imshow(error[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[0,i].set_title("t = {}".format(snapshots_index[i]*0.1))

    for i in range(6):
        for j in range(4):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    for i in range(6):    
        row_titles = ["ground truth", "full pred", "error", "FN0 1st lay", "FN0 2nd lay","ESN"]
        axs[i,0].set_ylabel(row_titles[i])


    plt.colorbar(c, ax=axs)
    fig.suptitle("U components of snapshots in training set")
    plt.show()








    fig, axs = plt.subplots(6,4, figsize=(15,12))

    snapshots_index = [0, 500, 1000, 1500]

    v_output = output_train[100:][...,1]
    v_fno_1 = preds_fno_1_train[100:][...,1]
    v_fno_2 = preds_fno_2_train[100:][...,1]
    v_esn = torch.tensor(output_esn_ps_train)[...,1]
    v_pred_train = full_pred_train[...,1]
    error = v_output - v_pred_train

    vmin = torch.min(torch.tensor([torch.min(v_fno_1), torch.min(v_fno_2), torch.min(v_esn), torch.min(v_output), torch.min(v_pred_train), torch.min(error)]))
    vmax = torch.max(torch.tensor([torch.max(v_fno_1), torch.max(v_fno_2), torch.max(v_esn),torch.max(v_output), torch.max(v_pred_train), torch.max(error)]))

    for i in range(4):
        axs[0,i].imshow(v_output[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[1,i].imshow(v_pred_train[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[3,i].imshow(v_fno_1[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[4,i].imshow(v_fno_2[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[5,i].imshow(v_esn[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        c = axs[2,i].imshow(error[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[0,i].set_title("t = {}".format(snapshots_index[i]*0.1))

    for i in range(6):
        for j in range(4):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    for i in range(6):    
        row_titles = ["ground truth", "full pred", "error", "FN0 1st lay", "FN0 2nd lay", "ESN"]
        axs[i,0].set_ylabel(row_titles[i])


    plt.colorbar(c, ax=axs)
    fig.suptitle("V components of snapshots in training set")
    plt.show()








    fig, axs = plt.subplots(6,4, figsize=(15,12))

    snapshots_index = [0, 500, 1000, 1500]

    u_output = output_test[100:][...,0]
    u_fno_1 = preds_fno_1_test[100:][...,0]
    u_fno_2 = preds_fno_2_test[100:][...,0]
    u_esn = torch.tensor(output_esn_ps_test)[...,0]
    u_pred_test = full_pred_test[...,0]
    error = u_output - u_pred_test

    vmin = torch.min(torch.tensor([torch.min(u_fno_1),torch.min(u_fno_2), torch.min(u_esn), torch.min(u_output), torch.min(u_pred_test), torch.min(error)]))
    vmax = torch.max(torch.tensor([torch.max(u_fno_1),torch.max(u_fno_2), torch.max(u_esn),torch.max(u_output), torch.max(u_pred_test), torch.max(error)]))

    for i in range(4):
        axs[0,i].imshow(u_output[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[1,i].imshow(u_pred_test[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[3,i].imshow(u_fno_1[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[4,i].imshow(u_fno_2[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[5,i].imshow(u_esn[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        c = axs[2,i].imshow(error[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[0,i].set_title("t = {}".format(snapshots_index[i]*0.1))

    for i in range(6):
        for j in range(4):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    for i in range(6):    
        row_titles = ["ground truth", "full pred", "error", "FN0 1st lay", "FN0 2nd lay", "ESN"]
        axs[i,0].set_ylabel(row_titles[i])


    plt.colorbar(c, ax=axs)
    fig.suptitle("U components of snapshots in testing set")
    plt.show()


    fig, axs = plt.subplots(6,4, figsize=(15,12))

    snapshots_index = [0, 500, 1000, 1500]

    v_output = output_test[100:][...,1]
    v_fno_1 = preds_fno_1_test[100:][...,1]
    v_fno_2 = preds_fno_2_test[100:][...,1]
    v_esn = torch.tensor(output_esn_ps_test)[...,1]
    v_pred_test = full_pred_test[...,1]
    error = v_output - v_pred_test

    vmin = torch.min(torch.tensor([torch.min(v_fno_1), torch.min(v_fno_2),torch.min(v_esn), torch.min(v_output), torch.min(v_pred_test), torch.min(error)]))
    vmax = torch.max(torch.tensor([torch.max(v_fno_1), torch.max(v_fno_2), torch.max(v_esn),torch.max(v_output), torch.max(v_pred_test), torch.max(error)]))

    for i in range(4):
        axs[0,i].imshow(v_output[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[1,i].imshow(v_pred_test[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[3,i].imshow(v_fno_1[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[4,i].imshow(v_fno_2[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[5,i].imshow(v_esn[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        c = axs[2,i].imshow(error[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[0,i].set_title("t = {}".format(snapshots_index[i]*0.1))

    for i in range(6):
        for j in range(4):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    for i in range(6):    
        row_titles = ["ground truth", "full pred", "error", "FN0 1st lay", "FN0 2nd lay", "ESN"]
        axs[i,0].set_ylabel(row_titles[i])


    plt.colorbar(c, ax=axs)
    fig.suptitle("V components of snapshots in testing set")
    plt.show()








if len(fno_model.modes)==1:
    output_esn_train = np.zeros((n_train-washout[0], resolution, resolution//2+1, 2), dtype=np.cfloat)
    output_esn_test = np.zeros((n_test-washout[0], resolution, resolution//2+1, 2), dtype=np.cfloat)

    pred_esn_train = esn_model.predict(W_input_train)
    pred_esn_test = esn_model.predict(W_input_test)
    pred_esn_train_rs = pred_esn_train.reshape(pred_esn_train.shape[0],pred_esn_train.shape[-1]//2,2)
    pred_esn_train_modes = pred_esn_train_rs[...,0] + 1j*pred_esn_train_rs[...,1]

    pred_esn_test_rs = pred_esn_test.reshape(pred_esn_test.shape[0],pred_esn_train.shape[-1]//2,2)
    pred_esn_test_modes = pred_esn_test_rs[...,0] + 1j*pred_esn_test_rs[...,1]


    kxx = np.fft.fftfreq(resolution, d=1.0) 
    kyy = np.fft.fftfreq(resolution, d=1.0)  
    kx, ky = np.meshgrid(kxx, kyy, indexing='ij')
    k_magnitude = np.sqrt(kx**2 + ky**2) 
    rk_magnitude = k_magnitude[:,:resolution//2+1]

    mask = np.logical_and(rk_magnitude <= k_end_esn, rk_magnitude >= k_start_esn)
    output_esn_train = np.zeros((n_train-washout[0], resolution, resolution//2+1, 2), dtype=np.cfloat)
    output_esn_test = np.zeros((n_test-washout[0], resolution, resolution//2+1, 2), dtype=np.cfloat)

    pred_esn_train = esn_model.predict(W_input_train)
    pred_esn_test = esn_model.predict(W_input_test)
    pred_esn_train_rs = pred_esn_train.reshape(pred_esn_train.shape[0],pred_esn_train.shape[-1]//4,2,2)
    pred_esn_train_modes = pred_esn_train_rs[...,0] + 1j*pred_esn_train_rs[...,1]

    pred_esn_test_rs = pred_esn_test.reshape(pred_esn_test.shape[0],pred_esn_train.shape[-1]//4,2,2)
    pred_esn_test_modes = pred_esn_test_rs[...,0] + 1j*pred_esn_test_rs[...,1]


    kxx = np.fft.fftfreq(resolution, d=1.0) 
    kyy = np.fft.fftfreq(resolution, d=1.0)  
    kx, ky = np.meshgrid(kxx, kyy, indexing='ij')
    k_magnitude = np.sqrt(kx**2 + ky**2) 
    rk_magnitude = k_magnitude[:,:resolution//2+1]

    mask = np.logical_and(rk_magnitude <= k_end_esn, rk_magnitude >= k_start_esn)
    #mask = rk_magnitude <= k_end_W

    output_esn_train[:,mask,:] = pred_esn_train_modes
    output_esn_test[:,mask,:] = pred_esn_test_modes

    output_esn_ps_train = np.fft.irfft2(output_esn_train, axes=(1,2))
    output_esn_ps_test = np.fft.irfft2(output_esn_test, axes=(1,2))
    full_pred_train = preds_fno_train[100:] + output_esn_ps_train
    full_pred_test = preds_fno_test[100:] + output_esn_ps_test


    preds_fno_train = fno_model(input_train, indexes=[0])
    preds_fno_test = fno_model(input_test, indexes=[0])

    error_train = output_train[100:] - full_pred_train
    error_test = output_test[100:] - full_pred_test


    save=False
    if not os.path.exists(os.path.join("./plot_models",model_name)):
        os.mkdir(os.path.join("./plot_models",model_name))
        save_path = os.path.join("./plot_models", model_name)
        save = True


    ### PLOT RESULTS ###
    def compute_classical_energy_spectrum(u):
        """
        Spectre classique par FFT directe des composantes de vitesse
        u : ndarray (t, nx, ny, 2)
        Returns: k_vals_classical, E_k_classical
        """
        tsteps, nx, ny, _ = u.shape
        E_k_sum = 0.0

        for t in range(tsteps):
            ux = u[t, :, :, 0]
            uy = u[t, :, :, 1]

            uxf = np.fft.fft2(ux)
            uyf = np.fft.fft2(uy)

            E2D = 0.5 * (np.abs(uxf)**2 + np.abs(uyf)**2) / (nx * ny)
            E2D = np.fft.fftshift(E2D)

            # Build radial wavenumber grid
            kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx
            ky = np.fft.fftshift(np.fft.fftfreq(ny)) * ny
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            k_mag = np.sqrt(KX**2 + KY**2)

            k_max = int(np.max(k_mag))
            E_k = np.zeros(k_max)
            for i in range(k_max):
                mask = (k_mag >= i) & (k_mag < i + 1)
                E_k[i] += np.sum(E2D[mask])

            E_k_sum += E_k

        E_k_classical = E_k_sum / tsteps
        k_vals_classical = np.arange(len(E_k_classical))
        return k_vals_classical, E_k_classical
        

    k_bin_centers, spectrum_binned_data = compute_classical_energy_spectrum(velocity_field)
    _, spectrum_binned_fno = compute_classical_energy_spectrum(preds_fno_train.detach().numpy())
    _, spectrum_binned_esn = compute_classical_energy_spectrum(output_esn_ps_train)
    _, spectrum_binned_full = compute_classical_energy_spectrum(full_pred_train.detach().numpy())

    cumulative_energy = np.cumsum(spectrum_binned_data)
    total_energy = cumulative_energy[-1]
    normalized_cumulative_energy = cumulative_energy / total_energy

    # Find the wavenumber where cumulative energy reaches 80%
    plt.figure(figsize=(10, 6))
    plt.loglog(k_bin_centers, spectrum_binned_fno, marker='o', linestyle='-', color='b', label='Energy Spectrum of FNO 1st lay. output')
    plt.loglog(k_bin_centers, spectrum_binned_esn, marker='o', linestyle='-', color='g', label='Energy Spectrum of ESN output')
    plt.loglog(k_bin_centers, spectrum_binned_full, marker='o', linestyle='-', color='black', label='Energy Spectrum of full output')
    plt.axvline(fno_model.modes[0], color='blue', linestyle='--', label=f'{np.round(normalized_cumulative_energy[fno_model.modes[0]-1]*100,2)}% Energy')
    plt.axvline(last_mode_ESN, color='purple', linestyle='--', label=f'{np.round(normalized_cumulative_energy[last_mode_ESN-1]*100,7)}% Energy')

    plt.loglog(k_bin_centers, spectrum_binned_data, marker='o', linestyle='-', color='r', label='Energy Spectrum of raw data')


    plt.ylim(1e-13,1e4)
    plt.xlabel('Wavenumber |k| (log scale)')
    plt.ylabel('Energy Spectrum E(k) (log scale)')
    plt.title('Energy Spectrum in Frequency Domain (Log-Log Plot)')
    plt.grid(True)
    plt.legend()
    plt.show()

    if save:
        plt.savefig(os.path.join(save_path, "energy_spectrum_train.png"))


    def compute_vorticity(vel_field):
        u = vel_field[..., 0]  # shape (1000, 64, 64)
        v = vel_field[..., 1]  # shape (1000, 64, 64)

        dv_dx = torch.gradient(v, dim=2)[0]
        du_dy = torch.gradient(u, dim=1)[0] 
        vorticity = dv_dx - du_dy
        return vorticity


    fig, axs = plt.subplots(5,4, figsize=(15,12))

    snapshots_index = [0, 500, 1000, 1500]

    vort_output = compute_vorticity(output_train[100:])
    vort_fno = compute_vorticity(preds_fno_train[100:])
    vort_esn = compute_vorticity(torch.tensor(output_esn_ps_train))
    vort_pred_train = compute_vorticity(full_pred_train)
    error = vort_output - vort_pred_train

    vmin = torch.min(torch.tensor([torch.min(vort_fno), torch.min(vort_esn), torch.min(vort_output), torch.min(vort_pred_train), torch.min(error)]))
    vmax = torch.max(torch.tensor([torch.max(vort_fno), torch.max(vort_esn), torch.max(vort_output), torch.max(vort_pred_train), torch.max(error)]))

    for i in range(4):
        axs[0,i].imshow(vort_output[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[1,i].imshow(vort_pred_train[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[3,i].imshow(vort_fno[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[4,i].imshow(vort_esn[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        c = axs[2,i].imshow(error[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[0,i].set_title("t = {}".format(snapshots_index[i]*0.1))

    for i in range(4):
        for j in range(4):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    for i in range(4):    
        row_titles = ["ground truth", "full pred", "error", "FN0 1st lay", "ESN"]
        axs[i,0].set_ylabel(row_titles[i])


    plt.colorbar(c, ax=axs)
    fig.suptitle("Vorticity of differents snapshots in training set")
    plt.show()

    if save:
        plt.savefig(os.path.join(save_path, "vorticity_train.png"))



    ###Test###

    k_bin_centers, spectrum_binned_data = compute_classical_energy_spectrum(velocity_field)
    _, spectrum_binned_fno = compute_classical_energy_spectrum(preds_fno_test.detach().numpy())
    _, spectrum_binned_esn = compute_classical_energy_spectrum(output_esn_ps_test)
    _, spectrum_binned_full = compute_classical_energy_spectrum(full_pred_test.detach().numpy())


    cumulative_energy = np.cumsum(spectrum_binned_data)
    total_energy = cumulative_energy[-1]
    normalized_cumulative_energy = cumulative_energy / total_energy

    # Find the wavenumber where cumulative energy reaches 80%
    plt.figure(figsize=(10, 6))
    plt.loglog(k_bin_centers, spectrum_binned_fno, marker='o', linestyle='-', color='b', label='Energy Spectrum of FNO 1st lay. output')
    plt.loglog(k_bin_centers, spectrum_binned_esn, marker='o', linestyle='-', color='g', label='Energy Spectrum of ESN output')
    plt.loglog(k_bin_centers, spectrum_binned_full, marker='o', linestyle='-', color='black', label='Energy Spectrum of full output')
    plt.axvline(fno_model.modes[0], color='blue', linestyle='--', label=f'{np.round(normalized_cumulative_energy[fno_model.modes[0]-1]*100,2)}% Energy')
    plt.axvline(last_mode_ESN, color='purple', linestyle='--', label=f'{np.round(normalized_cumulative_energy[last_mode_ESN-1]*100,7)}% Energy')

    plt.loglog(k_bin_centers, spectrum_binned_data, marker='o', linestyle='-', color='r', label='Energy Spectrum of raw data')


    plt.ylim(1e-13,1e4)
    plt.xlabel('Wavenumber |k| (log scale)')
    plt.ylabel('Energy Spectrum E(k) (log scale)')
    plt.title('Energy Spectrum in Frequency Domain (Log-Log Plot)')
    plt.grid(True)
    plt.legend()
    plt.show()

    if save:
        plt.savefig(os.path.join(save_path, "energy_spectrum_test.png"))


    def compute_vorticity(vel_field):
        u = vel_field[..., 0]  # shape (1000, 64, 64)
        v = vel_field[..., 1]  # shape (1000, 64, 64)

        dv_dx = torch.gradient(v, dim=2)[0]
        du_dy = torch.gradient(u, dim=1)[0] 
        vorticity = dv_dx - du_dy
        return vorticity


    fig, axs = plt.subplots(5,4, figsize=(15,12))

    snapshots_index = [0, 500, 1000, 1500]

    vort_output = compute_vorticity(output_test[100:])
    vort_fno = compute_vorticity(preds_fno_test[100:])
    vort_esn = compute_vorticity(torch.tensor(output_esn_ps_test))
    vort_pred_test = compute_vorticity(full_pred_test)
    error = vort_output - vort_pred_test

    vmin = torch.min(torch.tensor([torch.min(vort_fno), torch.min(vort_esn), torch.min(vort_output), torch.min(vort_pred_test), torch.min(error)]))
    vmax = torch.max(torch.tensor([torch.max(vort_fno), torch.max(vort_esn),torch.max(vort_output), torch.max(vort_pred_test), torch.max(error)]))

    for i in range(4):
        axs[0,i].imshow(vort_output[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[1,i].imshow(vort_pred_test[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[3,i].imshow(vort_fno[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[4,i].imshow(vort_esn[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        c = axs[2,i].imshow(error[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[0,i].set_title("t = {}".format(snapshots_index[i]*0.1))

    for i in range(4):
        for j in range(4):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    for i in range(4):    
        row_titles = ["ground truth", "full pred", "error", "FN0 1st lay", "ESN"]
        axs[i,0].set_ylabel(row_titles[i])


    plt.colorbar(c, ax=axs)
    fig.suptitle("Vorticity of differents snapshots in testing set")
    plt.show()


    if save:
        plt.savefig(os.path.join(save_path, "vorticity_test.png"))


    ### WASSERSTEIN DISTANCE ###

    import ot 

    def wass_dist(vel0,vel1):

        X = np.sqrt(vel0[...,0]**2 + vel0[...,1]**2)
        Y = np.sqrt(vel1[...,0]**2 + vel1[...,1]**2)

        a = np.ones(len(X)) / len(X)
        b = np.ones(len(Y)) / len(Y)

        M = ot.dist(X, Y)
        M /= M.max()  # optional normalization

        # Compute Wasserstein distance (order 1)
        d = ot.emd2(a, b, M) 

        return d


    wass_dist_train = np.zeros((n_train-washout[0],))
    wass_dist_test = np.zeros((n_test-washout[0],))[::10]

    print(full_pred_test.shape, output_test.shape, full_pred_train.dtype, full_pred_test.dtype, output_train.dtype, output_test.dtype)
    print("Compute wasserstein distance\n")
    print("Train...\n")
    for i in range(n_train-washout[0]):
        wass_dist_train[i] = wass_dist(full_pred_train[i].detach().numpy(), output_train[100:][i].detach().numpy())
    #print("Test...\n")
    for i in range(wass_dist_test.shape[0]):    
        wass_dist_test[i] = wass_dist(full_pred_test[10*i].detach().numpy(), output_test[100:][10*i].detach().numpy())


    print(wass_dist_train.shape, wass_dist_test.shape, wass_dist_train[0], wass_dist_test[0])
    # Downsample the data for plotting
    wass_dist_train_downsampled = wass_dist_train[::10]  # Take every 10th point
    wass_dist_test_downsampled = wass_dist_test


    # Calcul de l'erreur L2 pour le training set
    l2_error_train = np.zeros((n_train - washout[0],))
    for i in range(n_train - washout[0]):
        l2_error_train[i] = np.sqrt(np.mean((full_pred_train[i].detach().numpy() - output_train[100:][i].detach().numpy())**2))

    # Downsample les données pour le plot
    l2_error_train_downsampled = l2_error_train[::10]
    wass_dist_train_downsampled = wass_dist_train[::10]

    # Plot Wasserstein distance et erreur L2 pour le training set
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Courbe de la distance de Wasserstein
    ax1.plot(np.arange(len(wass_dist_train_downsampled)) * dt * 10, wass_dist_train_downsampled, label="Wasserstein Distance (Train)", color="blue")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Wasserstein Distance", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True)

    # Ajouter un deuxième axe pour l'erreur L2
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(l2_error_train_downsampled)) * dt * 10, l2_error_train_downsampled, label="L2 Error (Train)", color="red")
    ax2.set_ylabel("L2 Error", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    # Ajouter une légende combinée
    fig.legend(loc="upper right", bbox_to_anchor=(1, 0.9), bbox_transform=ax1.transAxes)

    plt.title("Wasserstein Distance and L2 Error Over Time (Train)")
    plt.show()

    if save:
        plt.savefig(os.path.join(save_path, "error_train.png"))

    # Calcul de l'erreur L2 pour le test set
    l2_error_test = np.zeros((n_test - washout[0],))
    for i in range(n_test - washout[0]):
        l2_error_test[i] = np.sqrt(np.mean((full_pred_test[i].detach().numpy() - output_test[100:][i].detach().numpy())**2))

    # Downsample les données pour le plot
    l2_error_test_downsampled = l2_error_test[::10]
    #wass_dist_test_downsampled = wass_dist_test[::10]

    # Plot Wasserstein distance et erreur L2 pour le test set
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Courbe de la distance de Wasserstein
    ax1.plot(np.arange(len(wass_dist_test_downsampled)) * dt * 10, wass_dist_test_downsampled, label="Wasserstein Distance (Test)", color="blue")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Wasserstein Distance", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")
    ax1.grid(True)

    # Ajouter un deuxième axe pour l'erreur L2
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(l2_error_test_downsampled)) * dt * 10, l2_error_test_downsampled, label="L2 Error (Test)", color="red")
    ax2.set_ylabel("L2 Error", color="red")
    ax2.tick_params(axis='y', labelcolor="red")

    # Ajouter une légende combinée
    fig.legend(loc="upper right", bbox_to_anchor=(1, 0.9), bbox_transform=ax1.transAxes)

    plt.title("Wasserstein Distance and L2 Error Over Time (Test)")
    plt.show()

    if save:
        plt.savefig(os.path.join(save_path, "error_test.png"))

    
    

    fig, axs = plt.subplots(5,4, figsize=(15,12))

    snapshots_index = [0, 500, 1000, 1500]

    u_output = output_train[100:][...,0]
    u_fno = preds_fno_train[100:][...,0]
    u_esn = torch.tensor(output_esn_ps_train)[...,0]
    u_pred_train = full_pred_train[...,0]
    error = u_output - u_pred_train

    vmin = torch.min(torch.tensor([torch.min(u_fno), torch.min(u_esn), torch.min(u_output), torch.min(u_pred_train), torch.min(error)]))
    vmax = torch.max(torch.tensor([torch.max(u_fno), torch.max(u_esn),torch.max(u_output), torch.max(u_pred_train), torch.max(error)]))

    for i in range(4):
        axs[0,i].imshow(u_output[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[1,i].imshow(u_pred_train[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[3,i].imshow(u_fno[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[4,i].imshow(u_esn[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        c = axs[2,i].imshow(error[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[0,i].set_title("t = {}".format(snapshots_index[i]*0.1))

    for i in range(4):
        for j in range(4):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    for i in range(4):    
        row_titles = ["ground truth", "full pred", "error", "FN0 1st lay", "ESN"]
        axs[i,0].set_ylabel(row_titles[i])


    plt.colorbar(c, ax=axs)
    fig.suptitle("U components of snapshots in training set")
    plt.show()








    fig, axs = plt.subplots(5,4, figsize=(15,12))

    snapshots_index = [0, 500, 1000, 1500]

    v_output = output_train[100:][...,1]
    v_fno = preds_fno_train[100:][...,1]
    v_esn = torch.tensor(output_esn_ps_train)[...,1]
    v_pred_train = full_pred_train[...,1]
    error = v_output - v_pred_train

    vmin = torch.min(torch.tensor([torch.min(v_fno), torch.min(v_esn), torch.min(v_output), torch.min(v_pred_train), torch.min(error)]))
    vmax = torch.max(torch.tensor([torch.max(v_fno), torch.max(v_esn),torch.max(v_output), torch.max(v_pred_train), torch.max(error)]))

    for i in range(4):
        axs[0,i].imshow(v_output[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[1,i].imshow(v_pred_train[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[3,i].imshow(v_fno[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[4,i].imshow(v_esn[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        c = axs[2,i].imshow(error[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[0,i].set_title("t = {}".format(snapshots_index[i]*0.1))

    for i in range(4):
        for j in range(4):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    for i in range(4):    
        row_titles = ["ground truth", "full pred", "error", "FN0 1st lay", "ESN"]
        axs[i,0].set_ylabel(row_titles[i])


    plt.colorbar(c, ax=axs)
    fig.suptitle("V components of snapshots in training set")
    plt.show()










    fig, axs = plt.subplots(5,4, figsize=(15,12))

    snapshots_index = [0, 500, 1000, 1500]

    u_output = output_test[100:][...,0]
    u_fno = preds_fno_test[100:][...,0]
    u_esn = torch.tensor(output_esn_ps_test)[...,0]
    u_pred_test = full_pred_test[...,0]
    error = u_output - u_pred_test

    vmin = torch.min(torch.tensor([torch.min(u_fno), torch.min(u_esn), torch.min(u_output), torch.min(u_pred_test), torch.min(error)]))
    vmax = torch.max(torch.tensor([torch.max(u_fno), torch.max(u_esn),torch.max(u_output), torch.max(u_pred_test), torch.max(error)]))

    for i in range(4):
        axs[0,i].imshow(u_output[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[1,i].imshow(u_pred_test[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[3,i].imshow(u_fno[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[4,i].imshow(u_esn[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        c = axs[2,i].imshow(error[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[0,i].set_title("t = {}".format(snapshots_index[i]*0.1))

    for i in range(4):
        for j in range(4):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    for i in range(4):    
        row_titles = ["ground truth", "full pred", "error", "FN0 1st lay", "ESN"]
        axs[i,0].set_ylabel(row_titles[i])


    plt.colorbar(c, ax=axs)
    fig.suptitle("U components of snapshots in testing set")
    plt.show()


    fig, axs = plt.subplots(5,4, figsize=(15,12))

    snapshots_index = [0, 500, 1000, 1500]

    v_output = output_test[100:][...,1]
    v_fno = preds_fno_test[100:][...,1]
    v_esn = torch.tensor(output_esn_ps_test)[...,1]
    v_pred_test = full_pred_test[...,1]
    error = v_output - v_pred_test

    vmin = torch.min(torch.tensor([torch.min(v_fno), torch.min(v_esn), torch.min(v_output), torch.min(v_pred_test), torch.min(error)]))
    vmax = torch.max(torch.tensor([torch.max(v_fno), torch.max(v_esn),torch.max(v_output), torch.max(v_pred_test), torch.max(error)]))

    for i in range(4):
        axs[0,i].imshow(v_output[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[1,i].imshow(v_pred_test[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[3,i].imshow(v_fno[snapshots_index[i]].detach().numpy(), vmin=vmin, vmax=vmax, cmap="bwr")
        axs[4,i].imshow(v_esn[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        c = axs[2,i].imshow(error[snapshots_index[i]], vmin=vmin, vmax=vmax, cmap="bwr")
        axs[0,i].set_title("t = {}".format(snapshots_index[i]*0.1))

    for i in range(4):
        for j in range(4):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

    for i in range(4):    
        row_titles = ["ground truth", "full pred", "error", "FN0 1st lay", "ESN"]
        axs[i,0].set_ylabel(row_titles[i])


    plt.colorbar(c, ax=axs)
    fig.suptitle("V components of snapshots in testing set")
    plt.show()