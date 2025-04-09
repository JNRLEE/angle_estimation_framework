import torch.optim as optim

OPTIMIZER_REGISTRY = {
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop,
    # Add more optimizers as needed
}

SCHEDULER_REGISTRY = {
    "StepLR": optim.lr_scheduler.StepLR,
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "ExponentialLR": optim.lr_scheduler.ExponentialLR,
    # Add more schedulers as needed
}

def get_optimizer(optimizer_name, params, **kwargs):
    """
    Retrieves an optimizer class from the registry and instantiates it.

    Args:
        optimizer_name (str): The name of the optimizer (must be in OPTIMIZER_REGISTRY).
        params (iterable): Model parameters to optimize.
        **kwargs: Additional keyword arguments to pass to the optimizer constructor (e.g., lr, weight_decay).

    Returns:
        torch.optim.Optimizer: An instantiated optimizer.
    """
    if optimizer_name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Optimizer '{optimizer_name}' not found. Available: {list(OPTIMIZER_REGISTRY.keys())}")

    optimizer_class = OPTIMIZER_REGISTRY[optimizer_name]
    try:
        # Filter out None values from kwargs, as some optimizers might not accept them
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return optimizer_class(params, **valid_kwargs)
    except Exception as e:
        raise RuntimeError(f"Error instantiating optimizer '{optimizer_name}' with args {valid_kwargs}: {e}")

def get_scheduler(scheduler_name, optimizer, **kwargs):
    """
    Retrieves a scheduler class from the registry and instantiates it.

    Args:
        scheduler_name (str): The name of the scheduler (must be in SCHEDULER_REGISTRY).
        optimizer (torch.optim.Optimizer): The optimizer instance.
        **kwargs: Additional keyword arguments to pass to the scheduler constructor (e.g., step_size, gamma, patience).

    Returns:
        torch.optim.lr_scheduler._LRScheduler: An instantiated scheduler, or None if scheduler_name is None or empty.
    """
    if not scheduler_name:
        return None # No scheduler requested

    if scheduler_name not in SCHEDULER_REGISTRY:
        raise ValueError(f"Scheduler '{scheduler_name}' not found. Available: {list(SCHEDULER_REGISTRY.keys())}")

    scheduler_class = SCHEDULER_REGISTRY[scheduler_name]
    try:
        # Filter out None values from kwargs
        valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return scheduler_class(optimizer, **valid_kwargs)
    except Exception as e:
        raise RuntimeError(f"Error instantiating scheduler '{scheduler_name}' with args {valid_kwargs}: {e}")

__all__ = ['get_optimizer', 'get_scheduler', 'OPTIMIZER_REGISTRY', 'SCHEDULER_REGISTRY'] 