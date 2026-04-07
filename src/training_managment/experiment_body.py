import os

from training_managment.trainer import Trainer
from training_managment.factory import Factory

import torch
import yaml

import time


DATA_DIR = os.getenv("DATA_DIR", "./data")
LOG_DIR = os.getenv("LOG_DIR", "../../runs")

class ExperimentBody():

    def __init__(self, args):

        self.device_asked = args.get("device", "auto")
        #Device parameters
        if self.device_asked in ["cuda","auto"] and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.device_asked in ["mps","auto"] and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif self.device_asked in ["cpu","auto"]:
            self.device = torch.device("cpu")

        print(f"Device used: {self.device}")
        
        self.args = args

        # File and repo managning 
        self.exp_dir = self.args["exp_dir"]
        self.exp_name = self.args["exp_name"]


    def make_directories(self):

        directories = [os.path.join(LOG_DIR,self.exp_dir),
                       os.path.join(LOG_DIR,self.exp_dir, self.exp_name),
                       os.path.join(LOG_DIR,self.exp_dir, self.exp_name, 'model_weights'),
                       os.path.join(LOG_DIR,self.exp_dir, self.exp_name, 'logs')]
        
        self.print_line()
        print("Creating directories...")
        for d in directories:
            os.makedirs(d, exist_ok=True)
            print(f"Directory created (or already existing): {d}")

        save_dir = os.path.join(LOG_DIR, self.exp_dir, self.exp_name)
        save_path = os.path.join(save_dir, "config.yaml")
        with open(save_path, "w") as f:
            yaml.safe_dump(self.args, f)  # écrit le dictionnaire args dans le fichier
            print(f"Configuration saved at: {save_path}")
        self.print_line()
        
    def print_line(self):
        print("-"*50)
