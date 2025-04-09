import torch
import torch.nn as nn

class OrderedMarginRankingLoss(nn.Module):
    """
    Calculates a loss combining a base regression loss (e.g., L1) with a margin ranking loss
    to enforce the correct order of predictions based on target angle values.
    Adapted from the Swin Transformer training script.
    """
    def __init__(self, angle_values, margin=1.0, base_loss_type='l1', base_loss_weight=1.0, ranking_loss_weight=1.0):
        """
        Args:
            angle_values (list or torch.Tensor): The ordered list of possible angle values.
            margin (float): The margin for the ranking loss.
            base_loss_type (str): Type of base loss for regression ('l1' or 'mse').
            base_loss_weight (float): Weight for the base regression loss component.
            ranking_loss_weight (float): Weight for the ranking loss component.
        """
        super().__init__()
        self.margin = margin
        self.angle_values = torch.tensor(angle_values, dtype=torch.float)
        self.base_loss_type = base_loss_type.lower()
        self.base_loss_weight = base_loss_weight
        self.ranking_loss_weight = ranking_loss_weight

        if self.base_loss_type == 'l1':
            self.base_loss_fn = nn.L1Loss(reduction='none') # Calculate per-element loss first
        elif self.base_loss_type == 'mse':
            self.base_loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unsupported base_loss_type: {base_loss_type}. Choose 'l1' or 'mse'.")

    def forward(self, outputs, target_indices):
        """
        Args:
            outputs (torch.Tensor): Model predictions, shape [batch_size, 1] or [batch_size].
            target_indices (torch.Tensor): Ground truth angle indices, shape [batch_size].
                                           Indices correspond to positions in `angle_values`.
        Returns:
            torch.Tensor: The combined scalar loss value.
        """
        batch_size = outputs.size(0)
        outputs = outputs.squeeze() # Ensure output is [batch_size]
        target_indices = target_indices.long() # Ensure indices are long

        if self.angle_values.device != outputs.device:
            self.angle_values = self.angle_values.to(outputs.device)

        # Get the actual target angle values corresponding to the indices
        target_angles = self.angle_values[target_indices]

        # 1. Calculate base regression loss (L1 or MSE)
        base_loss = self.base_loss_fn(outputs, target_angles)
        mean_base_loss = base_loss.mean()

        # 2. Calculate ranking loss constraint
        ranking_loss = torch.tensor(0.0, device=outputs.device)

        # Create all possible pairs within the batch
        # Use meshgrid for efficiency
        indices_i, indices_j = torch.meshgrid(torch.arange(batch_size, device=outputs.device),
                                              torch.arange(batch_size, device=outputs.device),
                                              indexing='ij')

        # Flatten to get all pairs (including self-pairs and duplicates, filter later)
        i_flat = indices_i.flatten()
        j_flat = indices_j.flatten()

        # Exclude self-pairs (i == j)
        non_self_mask = i_flat != j_flat
        i_valid = i_flat[non_self_mask]
        j_valid = j_flat[non_self_mask]

        # Get angles and outputs for valid pairs
        angles_i = target_angles[i_valid]
        angles_j = target_angles[j_valid]
        outputs_i = outputs[i_valid]
        outputs_j = outputs[j_valid]

        # Find pairs where angle_i > angle_j (these should also have output_i > output_j)
        order_mask = angles_i > angles_j

        if order_mask.sum() > 0:
            # Calculate margin loss for pairs where the order constraint should hold
            # We want output_i > output_j, so loss is max(0, margin - (output_i - output_j))
            pair_losses = torch.clamp(
                self.margin - (outputs_i[order_mask] - outputs_j[order_mask]),
                min=0.0
            )
            ranking_loss = pair_losses.mean()

        # Combine base loss and ranking loss
        total_loss = (self.base_loss_weight * mean_base_loss) + (self.ranking_loss_weight * ranking_loss)

        return total_loss

# You can also add the standard MarginRankingLoss here if needed,
# or simply use nn.MarginRankingLoss directly in the training script.
# Example wrapper if needed:
class StandardMarginRankingLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

    def forward(self, input1, input2, target):
        """
        Args:
            input1 (torch.Tensor): First input tensor (e.g., scores for item 1).
            input2 (torch.Tensor): Second input tensor (e.g., scores for item 2).
            target (torch.Tensor): Target tensor containing 1 or -1.
        """
        return self.loss_fn(input1, input2, target) 