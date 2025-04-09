import librosa
import numpy as np
import torch
import wave
import warnings
import os
from collections import Counter

def load_wav(file_path, target_sr=None, verbose=False):
    """
    Loads a WAV file and returns the audio signal and sample rate, with optional resampling.
    
    Description:
        Reads audio file using librosa, which supports various formats including WAV.
        If target_sr is provided, resamples the audio to the specified sample rate.
    
    Args:
        file_path (str): Path to the audio file to load.
        target_sr (int, optional): Target sample rate for resampling.
                                   If None, returns audio with original sample rate.
        verbose (bool, optional): Whether to print resampling information.
                                 Defaults to False to reduce terminal output.
    
    Returns:
        tuple: (audio_data, sample_rate) where:
            - audio_data (np.ndarray): Audio samples as a 1D numpy array, normalized to [-1, 1]
            - sample_rate (int): Sample rate of the returned audio
            
    References:
        - librosa.load: https://librosa.org/doc/main/generated/librosa.load.html
        - librosa.resample: https://librosa.org/doc/main/generated/librosa.resample.html
    """
    try:
        # 使用 librosa 讀取音頻檔案 (更健壯，支援重採樣)
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        
        # 檢查是否需要重採樣
        if target_sr is not None and sr != target_sr:
            # Only print resampling message if verbose is True
            if verbose:
                print(f"Resampling {file_path} from {sr} Hz to {target_sr} Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            
        return audio, sr
    except Exception as e:
        print(f"Error loading WAV file {file_path}: {e}")
        return None, None

def analyze_audio_sample_rates(file_paths, max_files=100):
    """
    Analyzes a list of audio files to report their original sample rates.
    
    Description:
        Scans a batch of audio files (or a subset if there are many) to detect
        their sample rates, which helps provide a summary of resampling needs
        rather than verbose per-file messages.
    
    Args:
        file_paths (list): List of paths to audio files
        max_files (int, optional): Maximum number of files to check if the list is large.
                                  Defaults to 100 to avoid long initialization times.
    
    Returns:
        dict: Information about sample rates in the dataset:
              - 'sample_rate_counts': Counter of original sample rates
              - 'needs_resampling': % of files needing resampling to 16kHz
              - 'files_checked': Number of files actually analyzed
    
    References:
        - librosa.get_samplerate: https://librosa.org/doc/main/generated/librosa.get_samplerate.html
    """
    if not file_paths:
        return {'sample_rate_counts': Counter(), 'needs_resampling': 0, 'files_checked': 0}
    
    # If too many files, check a representative sample
    files_to_check = file_paths
    if len(file_paths) > max_files:
        import random
        # Sample files evenly across the list to ensure representation
        step = len(file_paths) // max_files
        files_to_check = [file_paths[i] for i in range(0, len(file_paths), step)][:max_files]
    
    # Collect sample rates
    sample_rates = []
    for file_path in files_to_check:
        try:
            # Use librosa's get_samplerate function which is faster than full loading
            sr = librosa.get_samplerate(file_path)
            sample_rates.append(sr)
        except Exception as e:
            print(f"Error getting sample rate for {os.path.basename(file_path)}: {e}")
    
    # Count occurrences of each sample rate
    sr_counter = Counter(sample_rates)
    
    # Calculate how many files need resampling to 16kHz (common target)
    needs_resampling = sum(count for sr, count in sr_counter.items() if sr != 16000)
    pct_needs_resampling = (needs_resampling / len(sample_rates)) * 100 if sample_rates else 0
    
    result = {
        'sample_rate_counts': sr_counter,
        'needs_resampling': pct_needs_resampling,
        'files_checked': len(sample_rates)
    }
    
    return result

def compute_mel_spectrogram(audio, sample_rate, n_fft=1024, hop_length=256, n_mels=128, fmin=0, fmax=None):
    """
    Computes the Mel spectrogram for a given audio signal.
    
    Description:
        Converts raw audio waveform to a mel spectrogram representation,
        then converts to decibel scale, and adds a channel dimension.
    
    Args:
        audio (np.ndarray): Audio time series as a 1D numpy array.
        sample_rate (int): Sample rate of the audio.
        n_fft (int): FFT window size.
        hop_length (int): Number of samples between frames.
        n_mels (int): Number of Mel bands.
        fmin (float): Lowest frequency (in Hz).
        fmax (float): Highest frequency (in Hz). If None, defaults to sample_rate/2.
    
    Returns:
        torch.Tensor: Mel spectrogram with shape (1, n_mels, time_steps)
                     where time_steps depends on the audio length and hop_length.
                     Returns None if computation fails.
                     
    References:
        - librosa.feature.melspectrogram: https://librosa.org/doc/main/generated/librosa.feature.melspectrogram.html
        - librosa.power_to_db: https://librosa.org/doc/main/generated/librosa.power_to_db.html
    """
    if audio is None:
        return None
    try:
        # Suppress librosa warnings for specific operations if necessary
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mel_spect = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax if fmax is not None else sample_rate / 2
            )
        # Convert to decibels (log scale)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        # Add channel dimension (C, F, T)
        mel_spect_db = np.expand_dims(mel_spect_db, axis=0)
        return torch.FloatTensor(mel_spect_db)
    except Exception as e:
        print(f"Error computing Mel spectrogram: {e}")
        return None

def pad_or_truncate(spectrogram, target_length):
    """
    Pads or truncates the spectrogram along the time dimension.
    
    Description:
        Ensures that the time dimension (last dimension) of the spectrogram
        matches the target length by either truncating or zero-padding.
    
    Args:
        spectrogram (torch.Tensor): Input spectrogram tensor of shape (C, F, T)
                                   where T is the time dimension to modify.
        target_length (int): Target length for the time dimension.
    
    Returns:
        torch.Tensor: Modified spectrogram with shape (C, F, target_length).
                     Returns None if input is None.
                     
    References:
        - torch.cat: https://pytorch.org/docs/stable/generated/torch.cat.html
    """
    if spectrogram is None:
        return None
    current_length = spectrogram.shape[-1] # Time dimension is last
    if current_length == target_length:
        return spectrogram

    if current_length > target_length:
        return spectrogram[..., :target_length]
    else: # current_length < target_length
        padding_size = target_length - current_length
        # Pad shape: (0, 0) for time dim, (0, 0) for freq dim, (0, 0) for channel dim
        padding = (0, padding_size, 0, 0, 0, 0)
        # Need to convert tensor to numpy for padding potentially, then back
        # Or use torch.nn.functional.pad
        # Spectrogram shape is likely (C, F, T) -> (1, n_mels, current_length)
        pad_tensor = torch.zeros(
            spectrogram.shape[0], spectrogram.shape[1], padding_size,
            dtype=spectrogram.dtype, device=spectrogram.device
        )
        return torch.cat((spectrogram, pad_tensor), dim=-1)

def ensure_contiguous(tensor):
    """
    Ensures a tensor is contiguous in memory.
    
    Description:
        Checks if the tensor is already contiguous and returns it directly if so,
        otherwise creates a contiguous copy of the tensor.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: A tensor that is contiguous in memory.
        
    References:
        - torch.Tensor.is_contiguous: https://pytorch.org/docs/stable/generated/torch.Tensor.is_contiguous.html
        - torch.Tensor.contiguous: https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html
    """
    return tensor if tensor.is_contiguous() else tensor.contiguous() 