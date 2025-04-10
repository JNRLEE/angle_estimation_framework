from .datasets import AudioSpectrogramDataset
from .ranking import RankingPairDataset
from .dataset_utils import extract_metadata_from_filename, extract_angle_from_filename, filter_files_by_metadata

# Data Structure Information
"""
Current data structure:
/sinetone_link/step_018_sliced/
    /deg000/
        /box/
            box_deg000_500hz_00.wav
            box_deg000_1000hz_00.wav
            ...
        /plastic/
            plastic_deg000_500hz_00.wav
            plastic_deg000_1000hz_00.wav
            ...
    /deg018/
        /box/
            ...
        /plastic/
            ...
    ...
    /deg180/
        /box/
            ...
        /plastic/
            ...

Angle values: [000, 018, 036, 054, 072, 090, 108, 126, 144, 162, 180]
Materials: ['box', 'plastic']
Frequencies: [500, 1000, 3000]
"""

__all__ = ['AudioSpectrogramDataset', 'RankingPairDataset', 
           'extract_metadata_from_filename', 'extract_angle_from_filename', 
           'filter_files_by_metadata'] 