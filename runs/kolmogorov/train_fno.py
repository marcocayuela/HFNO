import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from training_managment.trainer import Trainer
from training_managment.experiment_body import ExperimentBody
from training_managment.factory import Factory
from utils.kolmogorov_loader import KolmogorovLoader



import models.hfno_2D as hfno
from torch.utils.data import DataLoader, TensorDataset #to manage datasets and bash 


class ExperimentKolmogorov(ExperimentBody):

    def __init__(self, args):
        super(ExperimentKolmogorov, self).__init__(args)

        self.args = args 
        print("Loading datasets...")
        self.datasets = KolmogorovLoader(self.args)
        print("Datasets loaded.")

        self.num_epochs = self.args["num_epochs"]   


    def execute_experience(self):

        print(f"Starting experiment: {self.exp_name}\n")
        self.make_directories()

        model = Factory.get_model(self.args["model"], self.args, self.device)
        model = model.to(self.device).float()    
        
        self.optimizer = Factory.get_optimizer(self.args["optimizer_info"]["type"], model.parameters(), lr=self.args["optimizer_info"]["lr"])
        self.scheduler = Factory.get_scheduler(self.args["scheduler_info"], self.optimizer, self.args["num_epochs"], self.datasets.n_batch_train)
        self.metrics = {metric: Factory.get_metric(metric) for metric in self.args["metrics_name"]}
        self.loss_fn = Factory.get_metric(self.args["loss_fn"])

        trainer = Trainer(model=model,
                          train_loader=self.datasets.training_loader,
                          test_loader=self.datasets.validation_loader,
                          loss_fn=self.loss_fn,
                          optimizer=self.optimizer,
                          scheduler=self.scheduler,
                          num_epochs=self.num_epochs,
                          device=self.device,
                          exp_dir=self.exp_dir,
                          exp_name=self.exp_name, 
                          metrics=self.metrics,
                          start_epoch=0)
        
        trainer.train_loop()


if __name__ == "__main__":

    args = {"exp_dir": "kolmogorov",
            "exp_name": "experiment1",
            "device": "cpu",
            "batch_size": 128,
            "p_val": 0.2,
            "time_ds": 2,
            "space_ds": 2,
            "model": "hfno_2d",
            "residual": "linear",
            "mode_separation": "L2",
            "layer_modes": [4,8],
            "depth": 32,
            "width_MLP": 32,
            "n_layers_MLP": 2,
            "input_size": 2,
            "output_size": 2,
            "res": 64,
            "optimizer_info": {"type": "adam", "lr": 1e-3},
            "scheduler_info": {"name": "cosine_annealing", "T_max": 500, "eta_min": 0.00001},
            "loss_fn": "mse",
            "metrics_name": ["mse", "rmse", "mae", "relative_rmse", "relative_mae"],
            "num_epochs": 500,
            }
    
    experiment = ExperimentKolmogorov(args)
    experiment.execute_experience()
