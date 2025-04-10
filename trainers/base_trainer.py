import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
import os
import json
from datetime import datetime
import numpy as np
import random

from data import AudioSpectrogramDataset, RankingPairDataset
from models import AngleEstimationModel
from losses import get_loss_function
from optimizers import get_optimizer, get_scheduler

# Helper function for calculating accuracy/metrics (example for regression -> classification)
def calculate_metrics(outputs, labels, angle_values, metric_prefix="", loss_value=None):
    """
    Calculates common metrics for evaluation.
    For regression, it can calculate MAE and accuracy by finding the closest angle class.
    For classification, it calculates accuracy directly from class predictions.
    
    Args:
        outputs (torch.Tensor): Model predictions (raw logits/values).
        labels (torch.Tensor): Ground truth labels (indices or values).
        angle_values (list or np.array): List of ordered angle values for classification mapping.
        metric_prefix (str): Prefix for metric names (e.g., 'val_', 'test_').
        loss_value (float, optional): Current batch/epoch loss to include in metrics.
    Returns:
        dict: Dictionary containing calculated metrics.
    """
    metrics = {}
    if loss_value is not None:
         metrics[f'{metric_prefix}loss'] = loss_value

    # 檢查輸出是一維(回歸)還是二維(分類)
    is_classification = len(outputs.shape) > 1 and outputs.shape[1] > 1
    
    if is_classification:
        # 分類任務 - 直接計算準確率
        outputs_detached = outputs.detach().cpu()
        labels_detached = labels.detach().cpu()
        
        # 獲取預測類別
        _, predicted = torch.max(outputs_detached, 1)
        
        # 計算準確率
        correct = (predicted == labels_detached).sum().item()
        total = labels_detached.size(0)
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        metrics[f'{metric_prefix}accuracy'] = accuracy
        metrics[f'{metric_prefix}correct'] = correct
        metrics[f'{metric_prefix}total'] = total
        
        # 如果有angle_values，還可以計算MAE
        if angle_values is not None:
            angle_tensor = torch.tensor(angle_values, dtype=torch.float)
            # 使用預測的類別索引和實際類別索引獲取對應角度
            predicted_angles = angle_tensor[predicted]
            actual_angles = angle_tensor[labels_detached.long()]
            mae = torch.abs(predicted_angles - actual_angles).mean().item()
            metrics[f'{metric_prefix}mae'] = mae
    else:
        # 回歸任務 - 原有的處理方式
        outputs = outputs.squeeze().detach().cpu()
        labels = labels.detach().cpu()

        if angle_values is not None:
            angle_tensor = torch.tensor(angle_values, dtype=torch.float)

            # Calculate MAE (Mean Absolute Error) - assumes labels are indices
            actual_angles = angle_tensor[labels.long()]
            mae = torch.abs(outputs - actual_angles).mean().item()
            metrics[f'{metric_prefix}mae'] = mae

            # Calculate Accuracy by mapping regression output to nearest angle class
            predicted_indices = []
            for pred_val in outputs:
                closest_idx = torch.argmin(torch.abs(angle_tensor - pred_val)).item()
                predicted_indices.append(closest_idx)
            predicted_indices = torch.tensor(predicted_indices)

            correct = (predicted_indices == labels.long()).sum().item()
            total = labels.size(0)
            accuracy = (correct / total) * 100 if total > 0 else 0
            metrics[f'{metric_prefix}accuracy'] = accuracy
            metrics[f'{metric_prefix}correct'] = correct
            metrics[f'{metric_prefix}total'] = total
        else:
            # Handle other cases, e.g., direct regression metrics if labels are values
            if outputs.shape == labels.shape:
                mae = torch.abs(outputs - labels).mean().item()
                metrics[f'{metric_prefix}mae'] = mae
            # Cannot calculate accuracy without defined classes/angle_values
            metrics[f'{metric_prefix}accuracy'] = -1 # Indicate not applicable

    return metrics


class BaseTrainer:
    """
    A base trainer class for handling training and evaluation loops.
    Configurable through a configuration dictionary.
    """
    def __init__(self, config, model=None, train_loader=None, val_loader=None, test_loader=None, device=None):
        """
        Args:
            config (dict): A dictionary containing configuration parameters for
                           data, model, loss, optimizer, scheduler, training, logging.
            model (torch.nn.Module, optional): 預先初始化的模型. 若為None, 將自動創建.
            train_loader (DataLoader, optional): 預先初始化的訓練數據加載器. 若為None, 將自動創建.
            val_loader (DataLoader, optional): 預先初始化的驗證數據加載器. 若為None, 將自動創建.
            test_loader (DataLoader, optional): 預先初始化的測試數據加載器. 若為None, 將自動創建.
            device (torch.device, optional): 運行設備. 若為None, 將根據配置或系統可用性選擇.
        """
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(config['logging']['log_dir'], f"run_{self.timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Results will be saved to: {self.results_dir}")

        # Process config to make sure numeric values are properly converted from scientific notation strings
        self._process_config_values(self.config)

        # Save config
        with open(os.path.join(self.results_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

        # 設置隨機種子以確保可重現性
        seed = config.get('seed', 42)
        self._set_random_seeds(seed)

        # Setup device
        self.device = device if device is not None else torch.device(config['training'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {self.device}")

        # Data related setup
        self.angle_values = config['data'].get('angle_values', None)

        # Training parameters
        self.epochs = config['training']['epochs']
        self.batch_size = config['training']['batch_size']
        self.use_ranking_loss = config['loss'].get('use_ranking_loss', False)
        self.gradient_clipping = config['training'].get('gradient_clipping', None)
        self.early_stopping_patience = config['training'].get('early_stopping_patience', None)
        self.early_stopping_metric = config['training'].get('early_stopping_metric', 'val_loss')
        self.early_stopping_mode = config['training'].get('early_stopping_mode', 'min') # min or max

        # Initialize state
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.best_metric_value = float('inf') if self.early_stopping_mode == 'min' else float('-inf')
        self.epochs_no_improve = 0
        self.history = {
            'train_loss': [], 'val_loss': [], 'test_loss': [],
            'val_accuracy': [], 'val_mae': [], 
            'test_accuracy': [], 'test_mae': [],
            'lr': []
        }

        # 根據提供的參數確定初始化流程
        self._setup_components(model, train_loader, val_loader, test_loader)

    def _set_random_seeds(self, seed):
        """
        設置隨機種子以確保實驗的可重現性
        
        Args:
            seed (int): 隨機種子值
            
        Returns:
            None
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

    def _setup_components(self, model, train_loader, val_loader, test_loader):
        """
        根據提供的組件選擇性初始化模型和數據加載器
        
        Args:
            model (torch.nn.Module, optional): 預先初始化的模型
            train_loader (DataLoader, optional): 預先初始化的訓練數據加載器
            val_loader (DataLoader, optional): 預先初始化的驗證數據加載器
            test_loader (DataLoader, optional): 預先初始化的測試數據加載器
            
        Returns:
            None
        """
        # 設置數據加載器(如果未提供)
        if train_loader is None or val_loader is None:
            self._setup_data()
        else:
            print("Using pre-initialized data loaders")
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader if test_loader is not None else None

        # 設置模型(如果未提供)
        if model is None:
            self._setup_model()
        else:
            print("Using pre-initialized model")
            self.model = model.to(self.device)
            self.model.print_trainable_parameters()

        # 無論是否已有模型，都需要設置損失函數、優化器和調度器
        self._setup_loss()
        self._setup_optimizer_scheduler()

    def _setup_data(self):
        """Sets up datasets and dataloaders."""
        print("--- Setting up Data --- ")
        data_config = self.config['data']
        base_dataset_config = {
            'audio_params': self.config.get('audio_params', {}),
            'data_filtering': data_config,
            'filename_pattern': data_config.get('filename_pattern', None),
            'label_mapping': data_config.get('label_mapping', None),
        }
        base_dataset = AudioSpectrogramDataset(data_config['data_dir'], base_dataset_config)

        if len(base_dataset) == 0:
            raise ValueError("Base dataset is empty after initialization. Check data path and filters.")

        # --- Split Dataset --- #
        train_split = data_config.get('train_split', 0.7)
        val_split = data_config.get('val_split', 0.15)
        # test_split = 1.0 - train_split - val_split # Implicit

        total_size = len(base_dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size

        if train_size == 0 or val_size == 0:
             raise ValueError(f"Calculated split sizes are too small (Train: {train_size}, Val: {val_size}). Adjust splits or check dataset size ({total_size}).")

        print(f"Splitting dataset ({total_size} samples): Train={train_size}, Val={val_size}, Test={test_size}")
        train_base_ds, val_base_ds, test_base_ds = torch.utils.data.random_split(base_dataset, [train_size, val_size, test_size])

        # --- Create Datasets for Training/Validation --- #
        if self.use_ranking_loss:
            print("Using RankingPairDataset for training and validation.")
            rank_config = self.config['loss'].get('ranking_params', {})
            self.train_dataset = RankingPairDataset(train_base_ds, num_pairs=rank_config.get('num_pairs_train', None))
            self.val_dataset = RankingPairDataset(val_base_ds, num_pairs=rank_config.get('num_pairs_val', None))
            if len(self.train_dataset) == 0 or len(self.val_dataset) == 0:
                 raise ValueError("RankingPairDataset resulted in zero pairs. Check base dataset and pair generation.")
        else:
            print("Using standard AudioSpectrogramDataset for training and validation.")
            self.train_dataset = train_base_ds
            self.val_dataset = val_base_ds

        # We might want a separate test dataset (not pairs) for final evaluation
        self.test_dataset = test_base_ds # Keep the base test dataset

        # --- Create DataLoaders --- #
        num_workers = self.config['training'].get('num_workers', 4)
        train_bs = min(self.batch_size, len(self.train_dataset)) if len(self.train_dataset) > 0 else 1
        val_bs = min(self.batch_size, len(self.val_dataset)) if len(self.val_dataset) > 0 else 1
        test_bs = min(self.batch_size, len(self.test_dataset)) if len(self.test_dataset) > 0 else 1

        self.train_loader = DataLoader(self.train_dataset, batch_size=train_bs, shuffle=True, num_workers=num_workers, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=val_bs, shuffle=False, num_workers=num_workers, drop_last=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=test_bs, shuffle=False, num_workers=num_workers, drop_last=False) if len(self.test_dataset) > 0 else None

        print(f"DataLoaders created: Train ({len(self.train_loader)} batches), Val ({len(self.val_loader)} batches), Test ({len(self.test_loader) if self.test_loader else 0} batches)")
        print("-------------------------")

    def _setup_model(self):
        """Sets up the model."""
        print("--- Setting up Model --- ")
        model_config = self.config['model']
        self.model = AngleEstimationModel(
            backbone_name=model_config['backbone']['name'],
            head_name=model_config['head']['name'],
            backbone_config=model_config['backbone'].get('params', {}),
            head_config=model_config['head'].get('params', {})
        )
        self.model.to(self.device)
        self.model.print_trainable_parameters()
        print("------------------------")

    def _setup_loss(self):
        """Sets up the loss function."""
        print("--- Setting up Loss --- ")
        loss_config = self.config['loss']
        loss_name = loss_config['name']
        loss_params = loss_config.get('params', {})

        # Add angle_values to params if required by the loss function
        if loss_name == 'OrderedMarginRankingLoss' and 'angle_values' not in loss_params:
             loss_params['angle_values'] = self.angle_values
             if self.angle_values is None:
                 raise ValueError("'angle_values' must be provided in data config for OrderedMarginRankingLoss.")

        self.criterion = get_loss_function(loss_name, **loss_params)
        print(f"Loss function: {loss_name} with params {loss_params}")
        print("-----------------------")

    def _setup_optimizer_scheduler(self):
        """Sets up the optimizer and scheduler."""
        print("--- Setting up Optimizer & Scheduler --- ")
        opt_config = self.config['optimizer']
        sched_config = self.config.get('scheduler', None)

        # Handle different learning rates for backbone and head if specified
        lr = opt_config['lr']
        # Convert learning rate to float if it's a string
        if isinstance(lr, str):
            lr = float(lr)
            
        backbone_lr = opt_config.get('backbone_lr', None)
        # Convert backbone learning rate to float if it's a string
        if isinstance(backbone_lr, str):
            backbone_lr = float(backbone_lr)
        
        params_to_optimize = []

        if backbone_lr is not None and backbone_lr != lr:
            print(f"Using different LRs: Backbone={backbone_lr}, Head={lr}")
            params_to_optimize = [
                {'params': self.model.backbone.parameters(), 'lr': backbone_lr},
                {'params': self.model.head.parameters(), 'lr': lr}
            ]
        else:
            print(f"Using single LR: {lr}")
            params_to_optimize = filter(lambda p: p.requires_grad, self.model.parameters())

        opt_name = opt_config['name']
        opt_params = opt_config.get('params', {})
        opt_params['lr'] = lr # Ensure base lr is passed

        self.optimizer = get_optimizer(opt_name, params_to_optimize, **opt_params)
        print(f"Optimizer: {opt_name} with params {opt_params}")

        if sched_config and sched_config.get('name'):
            sched_name = sched_config['name']
            sched_params = sched_config.get('params', {})
            
            # Convert string numeric parameters to float
            for key, value in sched_params.items():
                if isinstance(value, str) and (value.replace('.', '', 1).isdigit() or 
                                              (value.startswith('-') and value[1:].replace('.', '', 1).isdigit()) or
                                              ('e' in value.lower())):
                    try:
                        sched_params[key] = float(value)
                    except ValueError:
                        # Keep original value if conversion fails
                        pass
            
            self.scheduler = get_scheduler(sched_name, self.optimizer, **sched_params)
            print(f"Scheduler: {sched_name} with params {sched_params}")
        else:
            self.scheduler = None
            print("Scheduler: None")
        print("--------------------------------------")

    def _train_epoch(self, epoch):
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch_idx, batch_data in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # Move data to device
            if self.use_ranking_loss:
                # Expects (data1, data2, target)
                data1, data2, targets = batch_data
                data1, data2, targets = data1.to(self.device), data2.to(self.device), targets.to(self.device)
                outputs1 = self.model(data1)
                outputs2 = self.model(data2)
                loss = self.criterion(outputs1.squeeze(), outputs2.squeeze(), targets)
            else:
                # Expects (data, label_indices)
                data, labels = batch_data
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)

            loss.backward()

            # Gradient Clipping
            if self.gradient_clipping:
                clip_value = self.gradient_clipping.get('max_norm', 1.0)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_value)

            self.optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % self.config['logging'].get('log_interval', 50) == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(self.train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(self.train_loader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} Training Summary | Avg Loss: {avg_train_loss:.4f} | Time: {epoch_time:.2f}s")
        self.history['train_loss'].append(avg_train_loss)

    def _evaluate_epoch(self, epoch, data_loader, metric_prefix="val_"):
        """Runs a single evaluation epoch."""
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for batch_data in data_loader:
                if self.use_ranking_loss:
                    # Need to adapt evaluation for ranking loss, perhaps evaluate on base dataset?
                    # For now, we calculate loss on pairs, but other metrics might need base data.
                    data1, data2, targets = batch_data
                    data1, data2, targets = data1.to(self.device), data2.to(self.device), targets.to(self.device)
                    outputs1 = self.model(data1)
                    outputs2 = self.model(data2)
                    loss = self.criterion(outputs1.squeeze(), outputs2.squeeze(), targets)
                    total_loss += loss.item()
                    # Metrics like accuracy/MAE are harder to define directly on pairs
                    # We might need to evaluate on the underlying val_base_ds separately

                else:
                    data, labels = batch_data
                    data, labels = data.to(self.device), labels.to(self.device)
                    outputs = self.model(data)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item()
                    all_outputs.append(outputs.cpu())
                    all_labels.append(labels.cpu())

        avg_loss = total_loss / len(data_loader)
        metrics = {f'{metric_prefix}loss': avg_loss}

        if not self.use_ranking_loss and len(all_outputs) > 0:
            all_outputs_cat = torch.cat(all_outputs, dim=0)
            all_labels_cat = torch.cat(all_labels, dim=0)
            # Calculate metrics using the helper function
            eval_metrics = calculate_metrics(all_outputs_cat, all_labels_cat, self.angle_values, metric_prefix)
            metrics.update(eval_metrics)
            print(f"Epoch {epoch+1} {metric_prefix.capitalize()} Summary | Avg Loss: {avg_loss:.4f} | Accuracy: {metrics.get(f'{metric_prefix}accuracy', -1):.2f}% | MAE: {metrics.get(f'{metric_prefix}mae', -1):.4f}")
        else:
             # If using ranking loss, only loss is directly calculated here
             print(f"Epoch {epoch+1} {metric_prefix.capitalize()} Summary | Avg Loss: {avg_loss:.4f} (Metrics like Acc/MAE not calculated on pairs)")

        # Append to history (even if incomplete for ranking loss)
        self.history[f'{metric_prefix}loss'].append(avg_loss)
        self.history[f'{metric_prefix}accuracy'].append(metrics.get(f'{metric_prefix}accuracy', -1))
        self.history[f'{metric_prefix}mae'].append(metrics.get(f'{metric_prefix}mae', -1))

        return metrics # Return calculated metrics

    def _check_early_stopping(self, current_metrics, epoch):
        """Checks if early stopping criteria are met."""
        if not self.early_stopping_patience:
            return False # Early stopping not enabled

        metric_value = current_metrics.get(self.early_stopping_metric, None)
        if metric_value is None:
             print(f"Warning: Early stopping metric '{self.early_stopping_metric}' not found in evaluation results. Skipping check.")
             return False

        improved = False
        if self.early_stopping_mode == 'min':
            if metric_value < self.best_metric_value:
                self.best_metric_value = metric_value
                improved = True
        else: # mode == 'max'
            if metric_value > self.best_metric_value:
                self.best_metric_value = metric_value
                improved = True

        if improved:
            self.epochs_no_improve = 0
            print(f"Early stopping metric improved to {self.best_metric_value:.4f}. Saving best model.")
            self.save_checkpoint(epoch, is_best=True)
            return False # Continue training
        else:
            self.epochs_no_improve += 1
            print(f"Early stopping metric did not improve for {self.epochs_no_improve} epochs.")
            if self.epochs_no_improve >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                return True # Stop training
            return False

    def train(self):
        """Runs the main training loop."""
        print("\n=== Starting Training ===")
        start_train_time = time.time()

        for epoch in range(self.epochs):
            print(f"\n--- Epoch {epoch+1}/{self.epochs} --- ")
            self._train_epoch(epoch)
            val_metrics = self._evaluate_epoch(epoch, self.val_loader, metric_prefix="val_")

            # Append current learning rate to history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)

            # Scheduler Step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # Need validation metric for ReduceLROnPlateau
                    metric_for_scheduler = val_metrics.get(self.early_stopping_metric, None)
                    if metric_for_scheduler is not None:
                         self.scheduler.step(metric_for_scheduler)
                    else:
                         print(f"Warning: Metric '{self.early_stopping_metric}' needed for ReduceLROnPlateau scheduler not found.")
                else:
                    self.scheduler.step()

            # Save checkpoint (latest)
            self.save_checkpoint(epoch, is_best=False)

            # Check for early stopping
            if self._check_early_stopping(val_metrics, epoch):
                break

        total_train_time = time.time() - start_train_time
        print(f"\n=== Training Finished ({total_train_time:.2f}s) ===")
        self.save_history()

        # Load best model for final evaluation if early stopping was used
        if self.early_stopping_patience:
             print("Loading best model based on early stopping for final evaluation.")
             self.load_checkpoint(os.path.join(self.results_dir, 'model_best.pt'))

        # Final evaluation on the test set
        if self.test_loader:
             print("\n=== Evaluating on Test Set ===")
             test_metrics = self._evaluate_epoch(-1, self.test_loader, metric_prefix="test_") # Use -1 epoch for test
             print("Test Set Results:", test_metrics)
             # Save test metrics
             with open(os.path.join(self.results_dir, 'test_results.json'), 'w') as f:
                 json.dump(test_metrics, f, indent=4)
        else:
             print("No test set provided. Skipping final test evaluation.")

    def save_checkpoint(self, epoch, is_best=False):
        """Saves model checkpoint."""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric_value': self.best_metric_value,
            'config': self.config # Include config for reproducibility
        }
        filename = f"model_epoch_{epoch+1}.pt"
        filepath = os.path.join(self.results_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filepath}")

        if is_best:
            best_filepath = os.path.join(self.results_dir, "model_best.pt")
            torch.save(checkpoint, best_filepath)
            print(f"Saved best model checkpoint: {best_filepath}")

        # Keep only the last N checkpoints (optional)
        keep_last = self.config['logging'].get('keep_last_checkpoints', 5)
        if keep_last is not None and keep_last > 0:
             checkpoints = sorted(
                 [f for f in os.listdir(self.results_dir) if f.startswith('model_epoch_') and f.endswith('.pt')],
                 key=lambda x: int(x.split('_')[-1].split('.')[0])
             )
             if len(checkpoints) > keep_last:
                 for old_ckpt in checkpoints[:-keep_last]:
                     try:
                         os.remove(os.path.join(self.results_dir, old_ckpt))
                     except OSError as e:
                          print(f"Warning: Could not remove old checkpoint {old_ckpt}: {e}")

    def load_checkpoint(self, filepath):
        """Loads model checkpoint."""
        if not os.path.exists(filepath):
            print(f"Warning: Checkpoint file not found at {filepath}. Cannot load.")
            return

        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # Load optimizer and scheduler states if needed for resuming training
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # if self.scheduler and checkpoint['scheduler_state_dict']:
        #     self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # self.best_metric_value = checkpoint.get('best_metric_value', self.best_metric_value)
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Loaded checkpoint from {filepath} (Epoch {start_epoch})")

    def save_history(self):
        """Saves the training history to a JSON file."""
        history_path = os.path.join(self.results_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to {history_path}")

    def _process_config_values(self, config_dict):
        """
        Recursively process config dictionary to convert string representations of numbers to actual numeric types.
        
        Args:
            config_dict (dict): Configuration dictionary to process
            
        Returns:
            None (modifies the dictionary in-place)
        """
        if not isinstance(config_dict, dict):
            return
            
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self._process_config_values(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        self._process_config_values(item)
                    elif isinstance(item, str):
                        # Try to convert string to numeric if it looks like a number
                        try:
                            if 'e' in item.lower():  # Scientific notation
                                config_dict[key][i] = float(item)
                            elif '.' in item:  # Float
                                config_dict[key][i] = float(item)
                            elif item.lstrip('-').isdigit():  # Integer
                                config_dict[key][i] = int(item)
                        except ValueError:
                            pass  # Keep as string if conversion fails
            elif isinstance(value, str):
                # Try to convert string to numeric if it looks like a number
                try:
                    if 'e' in value.lower():  # Scientific notation
                        config_dict[key] = float(value)
                    elif '.' in value:  # Float
                        config_dict[key] = float(value)
                    elif value.lstrip('-').isdigit():  # Integer
                        config_dict[key] = int(value)
                except ValueError:
                    pass  # Keep as string if conversion fails 