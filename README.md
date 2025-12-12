# HFNO
Hierarchichal Fourier Neural Operator: an neural operator architecture inspired by FNO. 

Code will be available soon.


This repository contains the code to reproduce the experiments performed for the following paper: https://arxiv.org/abs/2511.01535.
3 experiments are conducted, one folder per experiment: "ks_equation", "kolmogorov_flow", "channel_flow".

The folder "data" is empty, you will find in the following the process to obtain the right datasets.
The folder "models" contains all the source code needed to execute experiments.


## Data generation

Kolmogorov flow: the datasets can be generated as we did with the following code: https://github.com/MagriLab/KolSol
KS_equation: we provide the file "" used to generate simulations in 1D for KS equation. To obtain different simulations, one can change the initial condition function into the file.
Channel_flow: we used channel flow data provided by Ardeshir Hanifi, Ricardo Vinuesa, and Arivazhagan Geetha Balasubramanian
from the Division of Fluid Mechanics, KTHRoyal Institute of Technology (Sweden). These data can be obtained on demand.


## Experiments 

### KS_equation
  Since the files are not so heavy, those corresponding to $\nu=0.7$ are available. There is thus nothing to generate to perform a test.

  To train a model: "python train_ks.py.
  Paramaters and hyperparameters can be changed in this file.

  To generate figure: "python results_generation.py".
  The name of the model to be test should be pu into this file.
  
### Kolmogorov flow
  This time the name of the .h5 simulation file should be change in train_fno.py
  First step: train the hfno without residual part with "python train_fno.py"
  Second step: train the ESN part with "python train_esn.py" 
  For each training, the hyperparameters can be changed in corresponding files.
  "python plot_results.py" generate figures associated to the selected model.


### Channel flow




