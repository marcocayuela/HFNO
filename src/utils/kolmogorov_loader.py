import h5py
import numpy as np
import torch
import os 

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import ConcatDataset

DATA_DIR = os.getenv("DATA_DIR", default="../../data")

class KolmogorovLoader():

    def __init__(self, args):

        self.batch_size = args["batch_size"]
        self.time_ds = args["time_ds"]
        self.space_ds = args["space_ds"]

        self.p_val = args["p_val"]

        self.traj_dir = os.path.join(DATA_DIR, f"kolmogorov/RE90/train_traj/")

        train_list = [1,2,3,4,5,6,7]
        test_list = [8]

        dataset_list_train = []
        dataset_list_val = []
        
        for i in train_list:
            sim_file = os.path.join(self.traj_dir, f"sim{i}.h5")
            if not os.path.isfile(sim_file):
                raise FileNotFoundError(f"Simulation file sim{i}.h5 not found in {self.traj_dir}. Please check the data directory and ensure the files are correctly named.")
            with h5py.File(sim_file, "r") as f:
                velocity_field = f["velocity_field"][()][::self.time_ds,::self.space_ds, ::self.space_ds]
            velocity_field = torch.from_numpy(velocity_field).type(torch.float32)
            n = velocity_field.shape[0]
            n_training = int(np.floor((1-self.p_val)*n))

            input_train = velocity_field[:n_training,...]
            output_train = velocity_field[1:(n_training+1)]
            input_val = velocity_field[n_training:-1]
            output_val = velocity_field[n_training+1:]

            dataset_list_train.append(TensorDataset(input_train, output_train))
            dataset_list_val.append(TensorDataset(input_val, output_val))  

        self.training_loader = DataLoader(ConcatDataset(dataset_list_train),
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    num_workers=0,
                                    persistent_workers=False,
                                    pin_memory=False)

        self.validation_loader = DataLoader(ConcatDataset(dataset_list_val),
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    persistent_workers=False,
                                    pin_memory=False)

        self.testing_traj = self.load_multiple_simulation(time_ds=self.time_ds, space_ds=self.space_ds, list=test_list)
                
        self.n_batch_train = len(self.training_loader)
        self.n_batch_validation = len(self.validation_loader)



    def load_simulation(self, data_path, time_ds=1, space_ds=1):
        '''
        Return the velocity field contained at the given file_path. It has to be a h5 file with the strcuture of the code KolSol
        Shape: [nt//time_ds, nx//space_ds, ny//space_ds,2]
        
        :param data_path: path where velocity field is contained
        :param time_ds: downsampling factor across time dimension
        :param space_ds: downsampling factore across space dimensions
        '''

        with h5py.File(data_path, "r") as f:
            velocity_field = f["velocity_field"][()][::time_ds,::space_ds, ::space_ds]
        return velocity_field


    def load_multiple_simulation(self, time_ds=1, space_ds=1, list=None):
        '''
        Load and return several simulations contain in the same repository. (They have to have same size)
        Shape: [n_simul, nt//time_ds, nx//space_ds, ny//space_ds,2]
        
        :param data_rep: Repository where simulations are saved.
        :param time_ds: downsampling factor across time dimension
        :param space_ds: downsampling factore across space dimensions
        :param list: (list of int) If given, load only simulations correpsonding to the given number
        '''

        data_rep = self.traj_dir
        if list is None:
            simulation_files = [os.path.join(data_rep, f) for f in os.listdir(data_rep) if f.endswith(".h5")]
        else:
            simulation_files = [os.path.join(data_rep, f)for f in os.listdir(data_rep) 
                                if f.startswith("sim") and f.endswith(".h5") and (int(f[3:-3]) in list)]
        
        simulation_velocities = []
        for file in simulation_files:
            with h5py.File(file, 'r') as f:
                simulation_velocities.append(f["velocity_field"][()][::time_ds,::space_ds, ::space_ds])
        
        return np.stack(simulation_velocities, axis=0)


    def get_info(data_path, print_info=True):

        '''
        Get the info and hyperparameters used for a kolmogorov simulation
        
        :param data_path: path where the simulation is located
        :param print: if true print the info
        '''

        infos = {}
        with h5py.File(data_path, "r") as f:
            infos["re"] = f["re"][()]
            infos["resolution"] = f["resolution"][()]
            infos["dt_simul"] = f["dt"][()]
            infos['simulation_time'] = f["time"][-1]
            infos["nt"] = f["velocity_field"][()].shape[0]
            infos["dt_saved"] = infos["simulation_time"]/infos["nt"]
            infos["nf"] = f["nf"][()]
            infos["nk"] = f["nk"][()]

        if print_info:
            for key, value in infos.items():
                print(key, ":", value)

        return infos


        