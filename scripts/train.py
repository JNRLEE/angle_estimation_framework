import argparse
import yaml
import os
import sys
import warnings
import torch
import random
import numpy as np
from datetime import datetime
import torch.nn as nn

# 忽略 Blowfish 演算法棄用警告
try:
    from cryptography.utils import CryptographyDeprecationWarning
    warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
except ImportError:
    # 如果 cryptography 未安裝，則跳過
    pass

# Add the project root to the Python path to allow relative imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the trainer
try:
    from trainers import BaseTrainer
except ImportError:
    print("Error: Could not import BaseTrainer. Please ensure you are running from the correct directory.")
    sys.exit(1)

def set_seed(seed=42):
    """
    Sets random seeds for reproducibility across different libraries.
    
    Description:
        Sets random seeds for Python's random module, NumPy, PyTorch, and
        CUDA if available, to ensure reproducible results across runs.
    
    Args:
        seed (int): The random seed value to use. Default is 42.
    
    Returns:
        None
    
    References:
        - PyTorch manual seed: https://pytorch.org/docs/stable/notes/randomness.html
        - NumPy random seed: https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility")

def load_config(config_path):
    """
    Loads and parses a YAML configuration file.
    
    Description:
        Reads a YAML file from the given path, handles potential errors,
        and returns the parsed configuration as a dictionary.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: The parsed configuration as a dictionary.
        
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If there's an error parsing the YAML file.
        RuntimeError: For other errors during file loading.
    
    References:
        - PyYAML documentation: https://pyyaml.org/wiki/PyYAMLDocumentation
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading config file {config_path}: {e}")

def main():
    """
    Main function to parse arguments, load configuration, and run training.
    
    Description:
        Parses command-line arguments, loads configuration from a YAML file,
        sets random seeds for reproducibility, initializes the trainer,
        and starts the training process.
    
    Args:
        None (uses command-line arguments)
    
    Returns:
        None
    
    References:
        - ArgParse module: https://docs.python.org/3/library/argparse.html
    """
    parser = argparse.ArgumentParser(description="Train an Angle Estimation Model")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    # Add other command-line overrides if needed (e.g., --epochs, --batch_size)
    # parser.add_argument("--epochs", type=int, help="Override number of epochs.")

    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)

    # Log experiment start
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"exp_{timestamp}_{os.path.basename(args.config).replace('.yaml', '')}"
    log_file = "experiments.log"
    
    with open(log_file, "a") as f:
        f.write(f"\n{experiment_id}: Started training with config {args.config}, seed={args.seed}\n")

    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        with open(log_file, "a") as f:
            f.write(f"{experiment_id}: Failed - {str(e)}\n")
        sys.exit(1)

    # --- Config Overrides (Example) ---
    # if args.epochs:
    #     config['training']['epochs'] = args.epochs
    #     print(f"Overriding epochs to: {args.epochs}")
    # ---------------------------------

    # --- Add Fine-tuning Logic --- (Needs refinement in BaseTrainer or here)
    unfreeze_epoch = config['training'].get('unfreeze_backbone_epoch')
    unfreeze_count = config['training'].get('unfreeze_layers_count')
    # Note: The current BaseTrainer doesn't automatically handle epoch-based unfreezing.
    # This logic would need to be integrated into the training loop, potentially
    # by adding a callback mechanism or modifying the _train_epoch method.
    # For now, the backbone unfreeze setting in the config applies from the start.
    if unfreeze_epoch is not None:
         print(f"Note: Epoch-based unfreezing (at epoch {unfreeze_epoch}) is configured but not yet implemented in BaseTrainer loop.")
         print("       The initial freezing state is determined by model.backbone.params.freeze_backbone/unfreeze_layers.")
    # -----------------------------

    # Initialize and run trainer
    try:
        trainer = BaseTrainer(config)
        trainer.train()
        print("Training completed successfully.")
        with open(log_file, "a") as f:
            f.write(f"{experiment_id}: Completed successfully\n")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        with open(log_file, "a") as f:
            f.write(f"{experiment_id}: Failed - {str(e)}\n")
        sys.exit(1)

if __name__ == "__main__":
    print("Executing main function...") # Add print statement here
    main() 