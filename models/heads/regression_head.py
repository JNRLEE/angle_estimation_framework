import torch
import torch.nn as nn

class RegressionHead(nn.Module):
    """
    A simple MLP head for regression tasks.
    Takes features from a backbone and predicts a single continuous value.
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout_rate=0.3, output_dim=1):
        """
        Args:
            input_dim (int): The dimensionality of the input features from the backbone.
            hidden_dims (list): A list of integers specifying the size of each hidden layer.
            dropout_rate (float): The dropout probability to apply after ReLU activations.
            output_dim (int): The dimensionality of the final output (default is 1 for single value regression).
        """
        super().__init__()
        layers = []
        last_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim

        # Final layer
        layers.append(nn.Linear(last_dim, output_dim))

        self.regressor = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features from the backbone (batch_size, input_dim).
        Returns:
            torch.Tensor: Regression output (batch_size, output_dim).
        """
        return self.regressor(x) 