import torch.nn as nn
from .ranking_losses import OrderedMarginRankingLoss, StandardMarginRankingLoss

# Dictionary to map loss names to their classes (including standard PyTorch losses)
LOSS_REGISTRY = {
    # Custom losses
    "OrderedMarginRankingLoss": OrderedMarginRankingLoss,
    "StandardMarginRankingLoss": StandardMarginRankingLoss,

    # Standard PyTorch losses (add more as needed)
    "MarginRankingLoss": nn.MarginRankingLoss,
    "L1Loss": nn.L1Loss,
    "MSELoss": nn.MSELoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss, # For classification tasks
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss, # For binary or multi-label classification
}

def get_loss_function(loss_name, **kwargs):
    """
    Retrieves a loss function class from the registry and instantiates it.

    Args:
        loss_name (str): The name of the loss function (must be in LOSS_REGISTRY).
        **kwargs: Additional keyword arguments to pass to the loss function's constructor.

    Returns:
        torch.nn.Module: An instantiated loss function.

    Raises:
        ValueError: If the loss_name is not found in the registry.
    """
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Loss function '{loss_name}' not found in registry. Available: {list(LOSS_REGISTRY.keys())}")

    loss_class = LOSS_REGISTRY[loss_name]
    try:
        return loss_class(**kwargs)
    except Exception as e:
        raise RuntimeError(f"Error instantiating loss function '{loss_name}' with args {kwargs}: {e}")

__all__ = ['get_loss_function', 'LOSS_REGISTRY', 'OrderedMarginRankingLoss', 'StandardMarginRankingLoss'] 