import torch
import sys
import os
import librosa
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones import SwinTransformerBackbone
from models.heads import RegressionHead
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

def test_model_fix(audio_path):
    """
    Test that the model fix works correctly by processing an audio file through the full model
    """
    print(f"Testing model fix with audio file: {audio_path}")
    
    # 1. Load audio and compute spectrogram
    print("\n1. Loading audio and computing spectrogram")
    waveform, sr = load_audio_file(audio_path, target_sr=16000)
    spectrogram = compute_spectrogram(waveform, n_fft=512, hop_length=256, normalize=True)
    
    # Add batch dimension
    spectrogram = spectrogram.unsqueeze(0)  # Now shape is [batch=1, channels=1, freq, time]
    print(f"Input spectrogram shape: {spectrogram.shape}")
    
    # 2. Create model
    print("\n2. Creating AngleEstimationModel")
    
    # Backbone config
    backbone_config = {
        "model_name": "microsoft/swin-tiny-patch4-window7-224",
        "pretrained": True,
        "use_feature_mapping": True,
        "freeze_backbone": True
    }
    
    # Head config
    head_config = {
        "input_dim": 512,
        "hidden_dims": [256, 128],
        "output_dim": 1,
        "dropout_rate": 0.2
    }
    
    # Create the model
    model = AngleEstimationModel(
        backbone_name="SwinTransformerBackbone",
        head_name="RegressionHead",
        backbone_config=backbone_config,
        head_config=head_config
    )
    
    # 3. Test forward pass
    print("\n3. Testing forward pass")
    try:
        with torch.no_grad():
            output = model(spectrogram)
            print(f"Model output shape: {output.shape}")
        print("\n✅ Success! The model forward pass works correctly with the fixed dimension adapter.")
        return True
    except Exception as e:
        print(f"\n❌ Error during model forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check if path provided as argument, otherwise use default
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "D:\\Transformer_training_Example\\sinetone_18degree_sliced\\data\\box_sinewave_deg000_500hz_00.wav"
    
    # Ensure the file exists
    if not os.path.exists(audio_path):
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)
    
    # Run the test
    success = test_model_fix(audio_path)
    sys.exit(0 if success else 1) 