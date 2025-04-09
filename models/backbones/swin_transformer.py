import torch
import torch.nn as nn
from transformers import SwinConfig, SwinModel, SwinForImageClassification
import time
import warnings

class FeatureMapping(nn.Module):
    """Maps input spectrogram (1 channel) to 3 channels expected by Swin."""
    def __init__(self, target_size=224):
        super().__init__()
        self.target_size = target_size
        # Simple mapping: Conv layers + Upsample
        self.mapping = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1), # Map to 3 channels
            nn.Upsample(size=(target_size, target_size), mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        # Input x shape: (batch, 1, freq, time)
        return self.mapping(x) # Output shape: (batch, 3, target_size, target_size)

class SwinTransformerBackbone(nn.Module):
    """
    Swin Transformer backbone using the Hugging Face transformers library.
    Loads a pre-trained Swin model and returns features before the final head.
    Includes optional feature mapping to adapt spectrogram input.
    """
    def __init__(self, model_name="microsoft/swin-tiny-patch4-window7-224",
                 pretrained=True, use_feature_mapping=True,
                 freeze_backbone=True, unfreeze_layers=0):
        """
        Args:
            model_name (str): The name of the pre-trained Swin model from Hugging Face Hub
                              or path to a local model.
            pretrained (bool): Whether to load pre-trained weights.
            use_feature_mapping (bool): Whether to include the FeatureMapping layer to adapt
                                        1-channel spectrograms to 3-channel input.
            freeze_backbone (bool): If True, freezes all layers of the Swin backbone initially.
            unfreeze_layers (int): Number of final layers (stages) to unfreeze if freeze_backbone is True.
                                   0 means keep all frozen, 1 unfreezes the last stage, etc.
        """
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.use_feature_mapping = use_feature_mapping
        self.target_size = 224 # Standard Swin input size

        if self.use_feature_mapping:
            self.feature_mapping = FeatureMapping(target_size=self.target_size)
        else:
            self.feature_mapping = nn.Identity() # Assumes input is already (B, 3, H, W)

        print(f"\n=== Initializing Swin Transformer Backbone ({model_name}) ===")
        try:
            start_time = time.time()
            # Load the base SwinModel (without the classification head)
            # Or load SwinForImageClassification and ignore/remove its head later
            # Using SwinModel is cleaner if we only need the backbone features.
            if self.pretrained:
                self.swin_model = SwinModel.from_pretrained(model_name)
                print(f"✓ Successfully loaded pre-trained SwinModel ({model_name}) in {time.time() - start_time:.2f}s")
            else:
                # Load config and initialize randomly
                config = SwinConfig.from_pretrained(model_name)
                self.swin_model = SwinModel(config)
                print(f"✓ Initialized SwinModel ({model_name}) with random weights in {time.time() - start_time:.2f}s")

            # Determine the output feature dimension
            # This is typically the dimensionality of the features after the final stage/pooling
            # For SwinModel, the output is `last_hidden_state` and `pooler_output`
            # `pooler_output` is usually suitable (shape: batch_size, hidden_size)
            self.output_dim = self.swin_model.config.hidden_size * (2**(len(self.swin_model.config.depths)-1)) # Heuristic for final stage dim
            if hasattr(self.swin_model, 'pooler') and hasattr(self.swin_model.pooler, 'dense'):
                 self.output_dim = self.swin_model.pooler.dense.out_features
            else:
                 warnings.warn("Could not automatically determine Swin output dimension from pooler. Using heuristic based on hidden_size and depths. Verify this is correct.")
            print(f"  - Backbone output dimension (pooler): {self.output_dim}")

        except Exception as e:
            print(f"X Error initializing SwinModel: {e}")
            raise

        print("=============================================================\n")

        # Handle freezing/unfreezing
        if freeze_backbone:
            self.freeze_all_layers()
            if unfreeze_layers > 0:
                self.unfreeze_last_stages(unfreeze_layers)
        else:
            print("Swin backbone initialized with all layers trainable.")


    def freeze_all_layers(self):
        """Freezes all parameters of the Swin backbone."""
        print("Freezing all Swin backbone layers.")
        for param in self.swin_model.parameters():
            param.requires_grad = False

    def unfreeze_last_stages(self, num_stages_to_unfreeze):
        """Unfreezes the last `num_stages_to_unfreeze` stages of the Swin model."""
        if num_stages_to_unfreeze <= 0:
            return

        total_stages = len(self.swin_model.encoder.layers)
        num_stages_to_unfreeze = min(num_stages_to_unfreeze, total_stages)

        print(f"Unfreezing the last {num_stages_to_unfreeze} stage(s) of Swin backbone...")

        # Unfreeze layers in the specified final stages
        for i in range(total_stages - num_stages_to_unfreeze, total_stages):
            print(f"  - Unfreezing Stage {i}")
            for param in self.swin_model.encoder.layers[i].parameters():
                param.requires_grad = True
            # Also unfreeze the downsampling layer associated with the start of this stage (if not the first stage)
            if i > 0 and self.swin_model.encoder.layers[i-1].downsample is not None:
                 print(f"    - Unfreezing Downsampler for Stage {i}")
                 for param in self.swin_model.encoder.layers[i-1].downsample.parameters():
                     param.requires_grad = True

        # Always unfreeze the final normalization and pooler layers
        if hasattr(self.swin_model, 'layernorm') and self.swin_model.layernorm is not None:
            print("  - Unfreezing Final LayerNorm")
            for param in self.swin_model.layernorm.parameters():
                param.requires_grad = True
        if hasattr(self.swin_model, 'pooler') and self.swin_model.pooler is not None:
            print("  - Unfreezing Pooler")
            for param in self.swin_model.pooler.parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor. Shape depends on `use_feature_mapping`.
                              If True: (batch, 1, freq, time)
                              If False: (batch, 3, height, width) - must match target_size.
        Returns:
            torch.Tensor: Output features from the Swin backbone's pooler (batch, output_dim).
        """
        x = self.feature_mapping(x)
        # Input x should now be (batch, 3, target_size, target_size)

        # Pass through Swin model
        outputs = self.swin_model(pixel_values=x)

        # Return the pooled output (features suitable for a classification/regression head)
        # Shape: (batch_size, hidden_size)
        pooled_output = outputs.pooler_output
        return pooled_output

    def get_output_dim(self):
        """Returns the output dimensionality of the backbone."""
        return self.output_dim

    def print_trainable_parameters(self):
        """Prints the number and percentage of trainable parameters."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
        # Optionally list trainable layers
        # print("Trainable layers:")
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"  - {name}") 