import torch
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../src'))

import models.hfno_1D
import models.fno_1D
import models.hfno_2D
import models.fno_2D

def relative_rmse(y_pred, y_true, eps=1e-8):
    """
    Computes Relative Root Mean Squared Error.
    Args:
        y_pred: Tensor of predictions
        y_true: Tensor of ground truth
        eps: small value to avoid division by zero
    Returns:
        scalar tensor
    """
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2))
    norm = torch.sqrt(torch.mean(y_true ** 2)) + eps
    return rmse / norm

def relative_mae(y_pred, y_true, eps=1e-8):
    """
    Computes Relative Root Mean Squared Error.
    Args:
        y_pred: Tensor of predictions
        y_true: Tensor of ground truth
        eps: small value to avoid division by zero
    Returns:
        scalar tensor
    """
    mae = torch.mean(torch.abs(y_pred - y_true))
    norm = torch.mean(torch.abs(y_true)) + eps
    return mae / norm

class Factory():

    OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    SCHEDULERS = {"one_cycle_lr": torch.optim.lr_scheduler.OneCycleLR, "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR}

    METRICS = {"mse": torch.nn.MSELoss(reduction='mean'),
               "rmse": lambda y_pred, y_true: torch.sqrt(torch.mean((y_pred - y_true)**2)),
               "mae": lambda y_pred, y_true: torch.mean(torch.abs(y_pred - y_true)),
               "relative_rmse": relative_rmse,
               "relative_mae": relative_mae}
    
    @staticmethod
    def get_optimizer(name, params, **kwargs):
        return Factory.OPTIMIZERS[name](params, **kwargs)
    
    @staticmethod
    def get_scheduler(scheduler_info, optimizer, num_epochs, n_batch):
        name = scheduler_info["name"]
        params = {k: v for k, v in scheduler_info.items() if k != "name"}
        if name == "one_cycle_lr":
            params["epochs"] = num_epochs
            params["steps_per_epoch"] = n_batch
        if name == "cosine_annealing":
            params["T_max"] = num_epochs

        return Factory.SCHEDULERS[name](optimizer, **params)

    @staticmethod
    def get_metric(name):
        return Factory.METRICS[name]
    
    @staticmethod
    def get_model(name, args, device, **kwargs):
        if name == "hfno_1d":
            return models.hfno_1D.HFNO_1D(args["layer_modes"], args["depth"], args["width_MLP"], args["n_layers_MLP"], args["input_size"], args["output_size"], device=device, **kwargs)
        
        if name == "fno_1d":
            return models.fno_1D.FNO1D(args["modes"], args["width"], args["l"], args["n_layer"], device=device, **kwargs)
        
        if name == "hfno_2d":
            return models.hfno_2D.HFNO_2D(modes=args["layer_modes"],
                                          depth=args["depth"],
                                          width_MLP=args["width_MLP"],
                                          n_layers_MLP=args["n_layers_MLP"],
                                          input_size=args["input_size"],
                                          output_size=args["output_size"],
                                          device=device,
                                          res=args["res"],
                                          residual=args["residual"],
                                          mode_separation=args["mode_separation"], **kwargs)
        
        if name == "fno_2d":
            return models.fno_2D.FNO2D(modes_x = args["modes"],
                                       modes_y = args["modes"],
                                       width=args["width"],
                                       l=args["l"],
                                       n_layer=args["n_layer"],
                                       device=device, **kwargs)