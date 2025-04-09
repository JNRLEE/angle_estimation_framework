import os
import sys
import torch
import glob

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset_utils import extract_metadata_from_filename, filter_files_by_metadata
from data.datasets import AudioSpectrogramDataset

def test_filename_extraction():
    """
    Test the filename metadata extraction on various filename patterns
    to ensure proper angle extraction
    """
    print("===== Testing filename metadata extraction =====")
    test_filenames = [
        "box_sinewave_deg000_500hz_00.wav",  # No index format
        "box_sinewave_deg018_1_500hz_00.wav",  # Index after deg format
        "box_sinewave_deg036_1_1000hz_01.wav",  # Index after deg with different Hz
        "box_sinewave_deg144_1_500hz_03.wav",  # Different angle
        "plastic_sinewave_deg162_3_500hz_08.wav",  # Different material
    ]
    
    print("\nTest extracting metadata from common filename patterns:")
    for filename in test_filenames:
        metadata = extract_metadata_from_filename(filename)
        print(f"File: {filename}")
        print(f"  - Material: {metadata.get('material', 'Not detected')}")
        print(f"  - Frequency: {metadata.get('frequency', 'Not detected')}")
        print(f"  - Index: {metadata.get('index', 'Not detected')}")
        print(f"  - Angle: {metadata.get('angle', 'Not detected')}")
        print()

def test_filter_by_metadata(data_dir):
    """
    Test the file filtering with various filter criteria
    """
    print("===== Testing file filtering with metadata =====")
    # Get all wav files in the directory
    search_pattern = os.path.join(data_dir, "*.wav")
    all_files = glob.glob(search_pattern, recursive=False)
    
    if not all_files:
        print(f"No wav files found in {data_dir}")
        return
    
    print(f"Found {len(all_files)} wav files in {data_dir}")
    
    # Test with various filter criteria
    filter_tests = [
        {
            "name": "All files (no filtering)",
            "criteria": {
                "material_filter": "all",
                "angle_values": [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
            }
        },
        {
            "name": "Only box material",
            "criteria": {
                "material_filter": "box",
                "angle_values": [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
            }
        },
        {
            "name": "Only 500Hz frequency",
            "criteria": {
                "material_filter": "all",
                "frequency_filter": [500],
                "angle_values": [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
            }
        },
        {
            "name": "Only indices 1-3",
            "criteria": {
                "material_filter": "all",
                "index_filter": [1, 2, 3],
                "angle_values": [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
            }
        }
    ]
    
    # Run each filter test
    for test in filter_tests:
        print(f"\nTest: {test['name']}")
        filtered_files, filtered_labels, stats = filter_files_by_metadata(all_files, test["criteria"])
        print(f"  - Accepted: {stats['accepted']} files")
        print(f"  - Rejected: {sum(stats['rejected'].values())} files")
        for reason, count in stats['rejected'].items():
            if count > 0:
                print(f"    - {reason.capitalize()}: {count}")
        
        # Print some sample files
        if filtered_files:
            print("\nSample accepted files:")
            for i in range(min(3, len(filtered_files))):
                metadata = extract_metadata_from_filename(filtered_files[i])
                angle = metadata.get('angle', 'Unknown')
                label = filtered_labels[i] if i < len(filtered_labels) else 'No label'
                print(f"  - {os.path.basename(filtered_files[i])} → Angle: {angle}, Label: {label}")

def test_audiospectrogram_dataset(data_dir):
    """
    Test the AudioSpectrogramDataset with the improved filename handling
    """
    print("\n===== Testing AudioSpectrogramDataset =====")
    
    # Create a config for the dataset
    config = {
        "audio_params": {
            "sample_rate": 16000,
            "n_fft": 1024,
            "hop_length": 256,
            "n_mels": 128,
            "target_length": 128
        },
        "data_filtering": {
            "material_filter": "all",
            "frequency_filter": [500, 1000, 3000],
            "index_filter": [1, 2, 3],
            "format_filter": ["wav"],
            "angle_values": [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]
        },
        "filename_pattern": "deg(\d+)"
    }
    
    try:
        # Create the dataset
        dataset = AudioSpectrogramDataset(data_dir, config)
        
        print(f"\nDataset contains {len(dataset)} samples")
        
        if len(dataset) > 0:
            # Test loading a sample
            print("\nTesting sample loading:")
            try:
                spectrogram, label = dataset[0]
                print(f"  - Successfully loaded sample")
                print(f"  - Spectrogram shape: {spectrogram.shape}")
                print(f"  - Label: {label}")
            except Exception as e:
                print(f"  - Error loading sample: {e}")
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Get data directory from command line or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "D:\\Transformer_training_Example\\sinetone_18degree_sliced\\data"
    
    # Ensure the directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        sys.exit(1)
    
    # Run the tests
    test_filename_extraction()
    test_filter_by_metadata(data_dir)
    test_audiospectrogram_dataset(data_dir) 