import torch
import torch.nn as nn
import warnings

# Dynamically get available backbones and heads
# This requires the __init__.py files to be set up correctly
from .backbones import * # Import all available backbones
from .heads import * # Import all available heads

class AngleEstimationModel(nn.Module):
    """
    A meta-model that combines a selected backbone and head for angle estimation.
    
    Description:
        This model provides a modular architecture for angle estimation from audio spectrograms,
        allowing easy configuration of different backbone networks (feature extractors) and
        head components (task-specific outputs). The backbone extracts features from the input,
        and the head processes these features to produce the final angle estimation.
    
    Args:
        backbone_name (str): The class name of the backbone to use (e.g., 'SimpleCNNBackbone', 'SwinTransformerBackbone').
        head_name (str): The class name of the head to use (e.g., 'RegressionHead').
        backbone_config (dict): Configuration dictionary for backbone constructor parameters.
                                May include keys like 'input_channels', 'output_feature_dim', etc.
        head_config (dict): Configuration dictionary for head constructor parameters.
                            May include keys like 'hidden_dims', 'dropout_rate', etc.
    
    Returns:
        AngleEstimationModel: An instantiated model combining the specified backbone and head.
        
    References:
        - PyTorch nn.Module: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
        - SimpleCNNBackbone: Custom CNN implementation for audio spectrograms
        - SwinTransformerBackbone: Adaptation of the Swin Transformer architecture for audio spectrograms
    """
    def __init__(self, backbone_name, head_name, backbone_config={}, head_config={}):
        """
        Initialize the AngleEstimationModel with specified backbone and head.
        
        Args:
            backbone_name (str): The class name of the backbone to use (e.g., 'SimpleCNNBackbone').
            head_name (str): The class name of the head to use (e.g., 'RegressionHead').
            backbone_config (dict): Configuration dictionary passed to the backbone constructor.
            head_config (dict): Configuration dictionary passed to the head constructor.
        """
        super().__init__()
        self.backbone_name = backbone_name
        self.head_name = head_name

        # Instantiate Backbone
        try:
            backbone_class = globals()[backbone_name]
            self.backbone = backbone_class(**backbone_config)
            print(f"Successfully instantiated backbone: {backbone_name}")
        except KeyError:
            raise ValueError(f"Backbone '{backbone_name}' not found. Available: {list(globals().keys())}")
        except Exception as e:
            raise RuntimeError(f"Error initializing backbone '{backbone_name}': {e}")

        # Get backbone output dimension - ALWAYS do a dummy forward pass for reliable dimension detection
        actual_output_dim = None
        try:
            # Standard input shape that works for most backbones: [batch=1, channels=1, freq=128, time=128]
            dummy_input = torch.randn(1, 1, 128, 128)
            # Move dummy input to the same device as the backbone
            if next(self.backbone.parameters(), None) is not None:
                dummy_input = dummy_input.to(next(self.backbone.parameters()).device)
            
            # Do a dummy forward pass to get actual output dimensions
            with torch.no_grad():
                dummy_output = self.backbone(dummy_input)
                actual_output_dim = dummy_output.shape[1]  # Get the actual feature dimension
                print(f"Detected actual backbone output dimension: {actual_output_dim}")
        except Exception as e:
            print(f"Warning: Error during dimension detection: {e}")
        
        # Use the backbone's reported dimension as fallback
        reported_dim = None
        if hasattr(self.backbone, 'get_output_dim'):
            reported_dim = self.backbone.get_output_dim()
            print(f"Backbone's reported output dimension: {reported_dim}")
        
        # Verify consistency between actual and reported dimensions
        if actual_output_dim is not None and reported_dim is not None and actual_output_dim != reported_dim:
            print(f"WARNING: Actual output dimension ({actual_output_dim}) does not match reported dimension ({reported_dim})")
            print(f"Using actual dimension ({actual_output_dim}) for dimension adapter")
            backbone_output_dim = actual_output_dim
        else:
            # Use whichever is available (actual preferred over reported)
            backbone_output_dim = actual_output_dim if actual_output_dim is not None else reported_dim
            
        if backbone_output_dim is None:
            raise RuntimeError(f"Could not determine backbone output dimension for {backbone_name}")

        # 添加維度轉換層，解決backbone和head維度不匹配問題
        target_dim = head_config.get('input_dim', 512)  # 預設目標維度為512
        
        if 'input_dim' not in head_config:
            head_config['input_dim'] = target_dim
            
        if backbone_output_dim != target_dim:
            print(f"Adding dimension adapter: {backbone_output_dim} -> {target_dim}")
            self.dim_adapter = nn.Sequential(
                nn.Linear(backbone_output_dim, target_dim),
                nn.ReLU()
            )
        else:
            self.dim_adapter = nn.Identity()
            print("No dimension adaptation needed")

        # Instantiate Head
        try:
            head_class = globals()[head_name]
            self.head = head_class(**head_config)
            print(f"Successfully instantiated head: {head_name}")
        except KeyError:
            raise ValueError(f"Head '{head_name}' not found. Available: {list(globals().keys())}")
        except Exception as e:
            raise RuntimeError(f"Error initializing head '{head_name}': {e}")

    def forward(self, x):
        """
        Forward pass through the backbone and head components.
        
        Description:
            Passes the input data through the backbone to extract features,
            then passes these features through the head to get the final output.
        
        Args:
            x (torch.Tensor): The input data (e.g., spectrogram) with appropriate shape
                              for the selected backbone.
        
        Returns:
            torch.Tensor: The final output from the head component, typically
                         representing an angle prediction or logits.
        
        References:
            - PyTorch forward method: https://pytorch.org/docs/stable/nn.html#torch.nn.Module.forward
        """
        features = self.backbone(x)
        features = self.dim_adapter(features)  # 應用維度轉換層
        output = self.head(features)
        return output

    # Optional: Add methods for freezing/unfreezing parts if needed
    def freeze_backbone(self):
        """
        Freezes all parameters of the backbone component.
        
        Description:
            Sets requires_grad=False for all parameters in the backbone
            to prevent them from being updated during training.
        
        Args:
            None
            
        Returns:
            None
            
        References:
            - PyTorch Parameter freezing: https://pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation
        """
        if hasattr(self.backbone, 'freeze_all_layers'):
            self.backbone.freeze_all_layers()
        else:
            print(f"Warning: Backbone {self.backbone_name} does not have freeze_all_layers method.")
            # Fallback: freeze all parameters directly
            for param in self.backbone.parameters():
                param.requires_grad = False

    def unfreeze_backbone_last_layers(self, num_layers):
        """
        Unfreezes the last few layers or stages of the backbone.
        
        Description:
            For transfer learning, often you want to keep early layers frozen
            and only fine-tune later layers. This method unfreezes a specified
            number of final layers/stages in the backbone.
            
        Args:
            num_layers (int): Number of layers/stages to unfreeze, counting from the end.
            
        Returns:
            None
            
        References:
            - Transfer learning guides: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        """
        if hasattr(self.backbone, 'unfreeze_last_stages'): # Specific to Swin in current impl
            self.backbone.unfreeze_last_stages(num_layers)
        else:
            print(f"Warning: Backbone {self.backbone_name} does not have specific unfreeze_last_stages method.")
            # More generic unfreezing could be added here if needed

    def print_trainable_parameters(self):
        """
        Prints statistics about trainable vs total parameters in the model.
        
        Description:
            Displays a summary of how many parameters are trainable in the backbone and head,
            which is useful for verifying that freezing/unfreezing has been applied correctly.
            
        Args:
            None
            
        Returns:
            None
            
        References:
            - PyTorch Parameter iteration: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters
        """
        print("--- Trainable Parameters --- ")
        if hasattr(self.backbone, 'print_trainable_parameters'):
            print("Backbone:")
            self.backbone.print_trainable_parameters()
        else:
            backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            backbone_total = sum(p.numel() for p in self.backbone.parameters())
            print(f"Backbone: {backbone_trainable:,}/{backbone_total:,}")

        head_trainable = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        head_total = sum(p.numel() for p in self.head.parameters())
        print(f"Head: {head_trainable:,}/{head_total:,}")
        print("---------------------------") 