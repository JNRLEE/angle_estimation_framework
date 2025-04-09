import torch
import torch.nn as nn

class SimpleCNNBackbone(nn.Module):
    """
    A simple CNN backbone for extracting features from spectrograms.
    Adapted from the feature extractor in the original simple_cnn_models_native.py.
    """
    def __init__(self, input_channels=1, output_feature_dim=256):
        """
        Args:
            input_channels (int): Number of input channels (e.g., 1 for grayscale spectrogram).
            output_feature_dim (int): The target dimensionality for the final layer before pooling.
                                        Note: The actual output dim after pooling will be this value.
        """
        super().__init__()
        self.output_feature_dim = output_feature_dim

        # CNN Feature Extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=(2, 4), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4 (adjust output channels to match output_feature_dim)
            nn.Conv2d(128, output_feature_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_feature_dim),
            nn.ReLU(inplace=True),
            # MaxPool2d removed here, using AdaptiveAvgPool2d after this block
        )

        # Adaptive Pooling to get a fixed-size output vector
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input spectrogram tensor (batch_size, channels, freq, time).
        Returns:
            torch.Tensor: Feature vector (batch_size, output_feature_dim).
        """
        # Input normalization (moved from adapter, can be done in dataset or here)
        x_mean = torch.mean(x, dim=[2, 3], keepdim=True)
        x_std = torch.std(x, dim=[2, 3], keepdim=True) + 1e-5 # Add epsilon for stability
        x = (x - x_mean) / x_std

        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) # Flatten the features
        return x

    def get_output_dim(self):
        """Returns the output dimensionality of the backbone."""
        return self.output_feature_dim 