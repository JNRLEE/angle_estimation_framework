"""
測試音頻讀取和重採樣功能
"""
import os
import sys

# 添加專案根目錄到 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.audio_utils import load_wav

def test_audio_resampling():
    """測試音頻檔案讀取和重採樣功能"""
    # 原始音頻檔案路徑
    test_audio_path = "D:/Transformer_training_Example/sinetone_18degree_sliced/data/box_sinewave_deg000_3000hz_02.wav"
    
    # 沒有重採樣
    print("測試不重採樣讀取...")
    audio, sr = load_wav(test_audio_path)
    print(f"原始採樣率: {sr} Hz")
    print(f"音頻長度: {len(audio)} 樣本")
    
    # 重採樣到 16000 Hz
    print("\n測試重採樣到 16000 Hz...")
    audio_resampled, sr_resampled = load_wav(test_audio_path, target_sr=16000)
    print(f"重採樣後採樣率: {sr_resampled} Hz")
    print(f"重採樣後音頻長度: {len(audio_resampled)} 樣本")
    
    # 驗證比例關係
    expected_length = len(audio) * (16000 / sr)
    print(f"期望的重採樣後長度: {expected_length:.2f} 樣本")
    print(f"實際長度與期望長度之比: {len(audio_resampled) / expected_length:.4f}")
    
    print("\n重採樣功能測試完成!")

if __name__ == "__main__":
    test_audio_resampling() 