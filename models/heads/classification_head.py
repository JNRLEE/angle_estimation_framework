import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    """
    A simple MLP head for classification tasks.
    Takes features from a backbone and predicts class probabilities.
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout_rate=0.3, num_classes=11):
        """
        Args:
            input_dim (int): The dimensionality of the input features from the backbone.
            hidden_dims (list): A list of integers specifying the size of each hidden layer.
            dropout_rate (float): The dropout probability to apply after ReLU activations.
            num_classes (int): The number of output classes (default is 11 for 0, 18, 36, ..., 180 degrees).
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

        # Final layer (no activation - will use CrossEntropyLoss which includes softmax)
        layers.append(nn.Linear(last_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features from the backbone (batch_size, input_dim).
        Returns:
            torch.Tensor: Classification logits (batch_size, num_classes).
        """
        return self.classifier(x) 