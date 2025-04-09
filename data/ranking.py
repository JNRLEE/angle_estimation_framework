import torch
from torch.utils.data import Dataset
import numpy as np

class RankingPairDataset(Dataset):
    """
    Creates ranking pairs from a base dataset for use with ranking losses.

    Generates pairs (data1, data2, target) where:
    - target = 1 if label(data1) > label(data2)
    - target = -1 if label(data1) < label(data2)
    Labels are assumed to be numerical for comparison.
    """
    def __init__(self, base_dataset, num_pairs=None, sampling_strategy='random'):
        """
        Args:
            base_dataset (Dataset): The underlying dataset from which to create pairs.
                                    Must return (data, label) where label is numerical.
            num_pairs (int, optional): The total number of pairs to generate.
                                     If None, generates N*(N-1) pairs where N is dataset size.
                                     Defaults to None.
            sampling_strategy (str): How to sample pairs ('random' or potentially 'all').
                                     Currently only 'random' is implemented efficiently.
        """
        self.base_dataset = base_dataset
        self.sampling_strategy = sampling_strategy
        self.num_base_samples = len(self.base_dataset)
        self.num_pairs = num_pairs

        # Pre-fetch labels for efficient pair generation if dataset is not huge
        # Consider lazy loading or alternative strategies for very large datasets
        print("Prefetching labels for ranking pair generation...")
        self.base_labels = [] # Store labels directly
        self.base_indices = [] # Store original indices
        for i in range(self.num_base_samples):
            try:
                _, label = self.base_dataset[i]
                # Ensure label is a comparable number (int/float)
                if isinstance(label, torch.Tensor):
                    label = label.item()
                self.base_labels.append(label)
                self.base_indices.append(i)
            except Exception as e:
                print(f"Warning: Could not retrieve item {i} from base dataset: {e}. Skipping.")
        print(f"Finished prefetching labels. Found {len(self.base_labels)} valid samples.")

        if len(self.base_labels) < 2:
            print("Warning: Base dataset has less than 2 valid samples. Cannot create pairs.")
            self.pairs = []
        else:
            self.pairs = self._create_ranking_pairs()

        print(f"Created {len(self.pairs)} ranking pairs.")

    def _create_ranking_pairs(self):
        pairs = []
        n_valid_samples = len(self.base_labels)

        # Determine the target number of pairs
        max_possible_pairs = n_valid_samples * (n_valid_samples - 1)
        target_num_pairs = self.num_pairs
        if target_num_pairs is None or target_num_pairs > max_possible_pairs:
            target_num_pairs = max_possible_pairs
            if self.num_pairs is not None:
                print(f"Warning: Requested num_pairs ({self.num_pairs}) > max possible ({max_possible_pairs}). Using max.")
        elif target_num_pairs <= 0:
             print(f"Warning: Requested num_pairs ({self.num_pairs}) is non-positive. Generating 0 pairs.")
             return []

        # Generate pairs using random sampling (more efficient for large N or subset)
        attempts = 0
        max_attempts = target_num_pairs * 5 # Heuristic to avoid infinite loop if few valid pairs exist

        while len(pairs) < target_num_pairs and attempts < max_attempts:
            idx1_internal, idx2_internal = np.random.choice(n_valid_samples, 2, replace=False)

            original_idx1 = self.base_indices[idx1_internal]
            original_idx2 = self.base_indices[idx2_internal]
            label1 = self.base_labels[idx1_internal]
            label2 = self.base_labels[idx2_internal]

            if label1 > label2:
                # Check for duplicates might be needed if true random sampling isn't strict enough
                # For now, assume duplicates are rare enough with large random sampling
                pairs.append((original_idx1, original_idx2, 1))
            elif label1 < label2:
                pairs.append((original_idx1, original_idx2, -1))
            # else: label1 == label2, skip this pair

            attempts += 1 # Increment attempt counter regardless of pair validity

        if len(pairs) < target_num_pairs:
            print(f"Warning: Could only generate {len(pairs)} unique pairs out of the target {target_num_pairs} after {attempts} attempts.")

        return pairs

    def __getitem__(self, idx):
        # Retrieve the indices and target for the pair
        original_idx1, original_idx2, target = self.pairs[idx]

        # Fetch the actual data from the base dataset using the original indices
        try:
            data1, label1 = self.base_dataset[original_idx1]
            data2, label2 = self.base_dataset[original_idx2]

            # Ensure target is a scalar float tensor
            target_tensor = torch.tensor(target, dtype=torch.float)

            # Return data pair, target, and original labels (optional, but useful for debugging/analysis)
            return data1, data2, target_tensor #, label1, label2 # Labels can be derived if needed

        except Exception as e:
            print(f"Error retrieving base data for pair indices ({original_idx1}, {original_idx2}) in __getitem__: {str(e)}")
            # Re-raise to let DataLoader handle skipping
            raise e

    def __len__(self):
        return len(self.pairs) 