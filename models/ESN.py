import torch
import numpy as np
from sklearn.linear_model import Ridge

class ESN:
    def __init__(self, input_size=128, hidden_size=1000, output_size=128, spectral_radius=0.9, alpha=1e-4):
        # Hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.alpha = alpha
        
        # Weights initialization
        self.W_in = torch.randn(hidden_size, input_size) * 0.01
        self.W_res = torch.randn(hidden_size, hidden_size)
        
        # Spectral radius adjustment
        self.W_res *= spectral_radius / torch.max(torch.abs(torch.linalg.eigvals(self.W_res)))
        
        self.readout = None

    def forward(self, input_seq, washout=100):
        """Compute reservoir states for the input sequence."""
        seq_len = input_seq.shape[0]
        hidden_states = torch.zeros(seq_len, self.hidden_size)
        hidden = torch.zeros(self.hidden_size)
        
        for t in range(seq_len):
            hidden = torch.tanh(self.W_in @ input_seq[t] + self.W_res @ hidden)
            hidden_states[t] = hidden
            
        return hidden_states[washout:]  # Ignore first 'washout' states

    def train(self, input_seq, target_seq, washout=100):
        """Training the readout layer using ridge regression."""
        # Forward pass
        reservoir_states = self.forward(input_seq, washout)  # [2300, hidden_size]
        targets = target_seq[washout:]  # [2300, 128]
        
        # Régression ridge
        self.readout = Ridge(alpha=self.alpha)
        self.readout.fit(reservoir_states.numpy(), targets.numpy())
        
    def predict(self, input_seq, washout=100):
        """Prediction"""
        reservoir_states = self.forward(input_seq, washout)
        return torch.tensor(self.readout.predict(reservoir_states.numpy()))

    def save(self, filepath):
        """Save the object to a file using torch."""
        torch.save({
            'W_in': self.W_in,
            'W_res': self.W_res,
            'readout_coef': self.readout.coef_ if self.readout else None,
            'readout_intercept': self.readout.intercept_ if self.readout else None,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'spectral_radius': self.spectral_radius
        }, filepath)

    @staticmethod
    def load(filepath):
        """Load the object from a file using torch."""
        checkpoint = torch.load(filepath)
        obj = ESN(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            output_size=checkpoint['output_size'],
            spectral_radius=checkpoint['spectral_radius']
        )
        obj.W_in = checkpoint['W_in']
        obj.W_res = checkpoint['W_res']
        if checkpoint['readout_coef'] is not None and checkpoint['readout_intercept'] is not None:
            obj.readout = Ridge()
            obj.readout.coef_ = checkpoint['readout_coef']
            obj.readout.intercept_ = checkpoint['readout_intercept']
        return obj