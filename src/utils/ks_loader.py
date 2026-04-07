import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import os 


DATA_DIR = os.getenv("DATA_DIR", "../../data")
print(DATA_DIR)
print(os.path.abspath(''))
class KSLoader():

    def __init__(self, args):


        self.m = args["m"]
        self.nu = int(100*args["nu"])
        self.batch_size = args["batch_size"]
        traj_dir = os.path.join(DATA_DIR, f"ks_equation/multitraj_nu{self.nu}")
        discard = 150

        # Lists to store chunks of trajectories
        train_data_list, train_deriv_list = [], []
        test_data_list, test_deriv_list = [], []

        # 1. Load Grid Characteristics (Assuming grid is shared and exists in the folder)
        grid_path = os.path.join(DATA_DIR, f"ks_equation/grid_{self.nu}.npy")
        grid_carac = torch.from_numpy(np.load(grid_path)).type(torch.float32)
        self.nt, self.dt, self.Tf, self.Mx, self.dx, self.L = grid_carac
        self.x_coordinates = torch.arange(0., self.Mx)*self.dx

        # 2. Iterate through all 10 trajectories (0 to 9)
        for i in range(10):
            # Construct filenames
            data_file = os.path.join(traj_dir, f"data_nu{self.nu}_id{i}.npy")
            deriv_file = os.path.join(traj_dir, f"mat_deriv_nu{self.nu}_id{i}.npy")
        
            # Load and discard first 150 snapshots (columns)
            # Shape is likely (Mx, Time), so we slice [:, discard:]
            d_raw = np.load(data_file)[:, discard:]
            m_raw = np.load(deriv_file)[:, discard:]
    
            d_tensor = torch.from_numpy(d_raw).type(torch.float32)
            m_tensor = torch.from_numpy(m_raw).type(torch.float32)
        
            # Sort into Train (0-7) or Test (8-9)
            if i < 8:
                train_data_list.append(d_tensor)
                train_deriv_list.append(m_tensor)
            else:
                test_data_list.append(d_tensor)
                test_deriv_list.append(m_tensor)

        # 3. Concatenate all trajectories along the time dimension
        all_train_data = torch.cat(train_data_list, dim=1)
        all_train_deriv = torch.cat(train_deriv_list, dim=1)

        all_test_data = torch.cat(test_data_list, dim=1)
        all_test_deriv = torch.cat(test_deriv_list, dim=1)

        # 4. Grid Encoding and Transposing
        # Training
        self.x_encoded, encoded_input_train = self.grid_encoding(all_train_data, encoding_dimension=self.m)
        self.input_train = encoded_input_train.T  # Shape (TotalTime, m)

        _, encoded_output_train = self.grid_encoding(all_train_deriv, encoding_dimension=self.m)
        self.output_train = encoded_output_train.T

        # Testing
        _, encoded_input_test = self.grid_encoding(all_test_data, encoding_dimension=self.m)
        self.input_test = encoded_input_test.T

        _, encoded_output_test = self.grid_encoding(all_test_deriv, encoding_dimension=self.m)
        self.output_test = encoded_output_test.T

        # 5. Final Setup
        self.n_train = self.input_train.shape[0]
        self.n_test = self.input_test.shape[0]

        self.training_set = DataLoader(
            TensorDataset(self.input_train, self.output_train), 
            batch_size=self.batch_size, 
            shuffle=True
        )

        self.testing_set = DataLoader(
            TensorDataset(self.input_test, self.output_test), 
            batch_size=self.n_test, 
            shuffle=False
        )

        self.n_batch_train = len(self.training_set)
        self.n_batch_test = len(self.testing_set)

    def grid_encoding(self, functions, encoding_dimension):

        P = functions.shape[0]
        M = functions.shape[1]
        
        step = P // encoding_dimension
        idx = np.arange(0, P, step)
        encoded_functions = functions[idx, :]
        encoding_coordinates = self.x_coordinates[idx]
        
        return encoding_coordinates, encoded_functions
