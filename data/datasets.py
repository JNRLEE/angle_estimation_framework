import os
import re
import glob
import torch
from torch.utils.data import Dataset
import numpy as np

from utils.audio_utils import load_wav, compute_mel_spectrogram, pad_or_truncate, analyze_audio_sample_rates
from .dataset_utils import extract_metadata_from_filename, filter_files_by_metadata

class AudioSpectrogramDataset(Dataset):
    """
    Dataset for loading audio files, converting them to Mel spectrograms,
    and extracting labels based on filename patterns.
    Handles filtering and padding/truncation.
    """
    def __init__(self, data_dir, config):
        """
        Args:
            data_dir (str): Path to the root directory containing audio files.
            config (dict or object): Configuration object/dict containing parameters like:
                - audio_params (dict): n_fft, hop_length, n_mels, target_length, sample_rate
                - data_filtering (dict): material_filter, frequency_filter, index_filter, format_filter, angle_values
                - filename_pattern (str, optional): Regex pattern to extract info (e.g., angle) from filename.
                                                      Needs capturing groups. Example: r"deg(\d+)" for angle.
                - label_mapping (dict, optional): Maps extracted string labels (like angle) to numerical indices.
                                                    If None, uses the extracted value directly (assuming it's numeric).
        """
        self.data_dir = data_dir
        self.config = config
        self.audio_params = config.get('audio_params', {})
        self.filter_params = config.get('data_filtering', {})
        self.filename_pattern = config.get('filename_pattern', None)
        self.label_mapping = config.get('label_mapping', None)

        # Audio parameters
        self.sample_rate = self.audio_params.get('sample_rate', 16000) # Default if not provided
        self.n_fft = self.audio_params.get('n_fft', 1024)
        self.hop_length = self.audio_params.get('hop_length', 256)
        self.n_mels = self.audio_params.get('n_mels', 128)
        self.target_length = self.audio_params.get('target_length', 128) # Target time steps

        # Filtering parameters
        self.material_filter = self.filter_params.get('material_filter', "all")
        self.frequency_filter = self.filter_params.get('frequency_filter', None)
        self.index_filter = self.filter_params.get('index_filter', None)
        self.format_filter = self.filter_params.get('format_filter', ["wav"])
        self.angle_values = self.filter_params.get('angle_values', None) # e.g., [0, 18, 36, ...]

        # Ensure filters are lists
        if self.frequency_filter and not isinstance(self.frequency_filter, list):
            self.frequency_filter = [self.frequency_filter]
        if self.index_filter and not isinstance(self.index_filter, list):
            self.index_filter = [self.index_filter]
        if not isinstance(self.format_filter, list):
            self.format_filter = [self.format_filter]

        self.file_paths = []
        self.labels = []
        self._load_file_paths()

    def _load_file_paths(self):
        """Scans the data directory, parses filenames, applies filters, and stores valid paths and labels."""
        print(f"Loading audio data from: {self.data_dir}")
        print(f"Filtering settings: {self.filter_params}")
        
        if not os.path.exists(self.data_dir):
            print(f"Error: Data directory not found at {self.data_dir}")
            return

        # Create filter criteria dictionary
        filter_criteria = {
            'material_filter': self.material_filter,
            'frequency_filter': self.frequency_filter,
            'index_filter': self.index_filter,
            'angle_values': self.angle_values
        }
        
        # Find all files that match the format filter
        all_files = []
        for fmt in self.format_filter:
            search_pattern = os.path.join(self.data_dir, f"*.{fmt}")
            all_files.extend(glob.glob(search_pattern, recursive=False))
        
        if not all_files:
            print(f"Warning: No files found matching pattern {self.data_dir}/*.{self.format_filter}")
            return

        # Analyze sample rates before filtering to give summary information
        sr_analysis = analyze_audio_sample_rates(all_files)
        if sr_analysis['sample_rate_counts']:
            print(f"Audio sample rate analysis:")
            for sr, count in sr_analysis['sample_rate_counts'].items():
                print(f"  - {sr} Hz: {count} files")
            if sr_analysis['needs_resampling'] > 0:
                print(f"  - {sr_analysis['needs_resampling']:.1f}% of files will be resampled to {self.sample_rate} Hz")
                print(f"  - Resampling messages will be suppressed during data loading for cleaner output")
            print(f"  - Analysis based on {sr_analysis['files_checked']} out of {len(all_files)} files")

        # Apply the improved filtering
        filtered_files, filtered_labels, stats = filter_files_by_metadata(all_files, filter_criteria)
        
        # Store the filtered files and labels
        self.file_paths = filtered_files
        self.labels = filtered_labels
        
        # Print statistics
        print(f"File filtering completed:")
        print(f"  - Total files found: {stats['total']}")
        print(f"  - Files accepted: {stats['accepted']}")
        print(f"  - Files rejected due to:")
        for reason, count in stats['rejected'].items():
            if count > 0:
                print(f"    - {reason.capitalize()}: {count}")
        
        # Print angle distribution if available
        if self.angle_values and self.labels:
            angle_counts = {}
            for i, angle in enumerate(self.angle_values):
                count = self.labels.count(i)
                angle_counts[angle] = count
            print(f"Angle distribution: {angle_counts}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            audio, sr = load_wav(file_path, target_sr=self.sample_rate, verbose=False)
            if audio is None:
                # Handle inconsistent sample rates if necessary, or raise error
                print(f"Warning: Skipping {file_path}. Could not load audio file.")
                # Return dummy data or raise error? For now, raise to indicate issue.
                raise RuntimeError(f"Failed to load {file_path}.")

            mel_spectrogram = compute_mel_spectrogram(
                audio, self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
            )

            if mel_spectrogram is None:
                 raise RuntimeError(f"Failed to compute spectrogram for {file_path}.")

            # Pad or truncate
            processed_spectrogram = pad_or_truncate(mel_spectrogram, self.target_length)

            # Ensure contiguous tensor
            processed_spectrogram = processed_spectrogram.contiguous()

            # Return spectrogram and label (convert label to tensor)
            # Using float for label to support regression directly
            return processed_spectrogram, torch.tensor(label, dtype=torch.float)

        except Exception as e:
            print(f"Error processing file {file_path} in __getitem__: {str(e)}")
            # Depending on use case, might return None or a placeholder, or re-raise
            # Re-raising makes the DataLoader skip this sample if default collate_fn is used
            raise e 