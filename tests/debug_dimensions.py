import torch
import sys
import os
import librosa
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones import SwinTransformerBackbone
from models.meta_model import AngleEstimationModel

def load_audio_file(file_path, target_sr=16000):
    """Load audio file with optional resampling"""
    try:
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        
        # Optional resampling
        if target_sr is not None and sr != target_sr:
            print(f"Resampling {file_path} from {sr} Hz to {target_sr} Hz")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            
        return audio, sr
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        raise

def compute_spectrogram(audio, n_fft=1024, hop_length=256, normalize=True):
    """Compute spectrogram from audio data"""
    try:
        # Compute mel spectrogram
        mel_spect = librosa.feature.melspectrogram(
            y=audio, sr=16000, n_fft=n_fft, hop_length=hop_length, n_mels=128
        )
        
        # Convert to decibels
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        
        # Add channel dimension and convert to tensor
        mel_spect_db = np.expand_dims(mel_spect_db, axis=0)
        return torch.FloatTensor(mel_spect_db)
    except Exception as e:
        print(f"Error computing spectrogram: {e}")
        raise

def debug_dimensions(audio_path):
    """
    Debug the dimensions of tensors as they flow through the model pipeline.
    """
    print(f"Debug dimensions for file: {audio_path}")
    
    # 1. Load audio and compute spectrogram
    print("\n1. Loading audio and computing spectrogram")
    waveform, sr = load_audio_file(audio_path, target_sr=16000)
    spectrogram = compute_spectrogram(waveform, n_fft=512, hop_length=256, normalize=True)
    
    # Add batch dimension
    spectrogram = spectrogram.unsqueeze(0)  # Now shape is [batch=1, channels=1, freq, time]
    print(f"Input spectrogram shape: {spectrogram.shape}")
    
    # 2. Initialize the Swin backbone
    print("\n2. Initializing SwinTransformerBackbone")
    backbone = SwinTransformerBackbone(
        model_name="microsoft/swin-tiny-patch4-window7-224",
        pretrained=True,
        use_feature_mapping=True,
        freeze_backbone=True
    )
    
    print(f"Backbone output dimension: {backbone.get_output_dim()}")
    
    # 3. Check the FeatureMapping layer output
    print("\n3. Checking FeatureMapping output shape")
    with torch.no_grad():
        feature_mapped = backbone.feature_mapping(spectrogram)
        print(f"After feature mapping: {feature_mapped.shape}")
    
    # 4. Check backbone output
    print("\n4. Checking backbone output shape")
    with torch.no_grad():
        backbone_output = backbone(spectrogram)
        print(f"Backbone output shape: {backbone_output.shape}")
    
    # 5. Verify the dimension adapter
    print("\n5. Testing dimension adapter")
    if backbone_output.shape[1] != backbone.get_output_dim():
        print(f"WARNING: Output dimension mismatch!")
        print(f"Expected dimension: {backbone.get_output_dim()}")
        print(f"Actual dimension: {backbone_output.shape[1]}")
        
        # The error is: mat1 and mat2 shapes cannot be multiplied (32x768 and 6144x512)
        # This means backbone_output is [batch=32, features=768] 
        # But the Linear layer expects [batch, 6144] as input
        print("\nAnalyzing error: mat1 and mat2 shapes cannot be multiplied (32x768 and 6144x512)")
        print("- Mat1 shape: Actual backbone output shape is [batch, features]")
        print("- Mat2 shape: Linear weight matrix is [6144, 512]")
        print("- Conclusion: The backbone output dimension doesn't match what the Linear layer expects")
        
        # Possible solution: Reshape the output to match the expected input
        print("\nPossible solution: Create a proper dimension adapter")
    
    return backbone_output

if __name__ == "__main__":
    # Check if path provided as argument, otherwise use default
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "D:\\Transformer_training_Example\\sinetone_18degree_sliced\\data\\box_sinewave_deg000_500hz_00.wav"
    
    # Ensure the file exists
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)
    
    try:
        output = debug_dimensions(audio_path)
        print("\nDebug completed successfully.")
    except Exception as e:
        print(f"\nError during debugging: {e}")
        import traceback
        traceback.print_exc() 