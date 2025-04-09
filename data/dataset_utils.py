import re
import os

def extract_angle_from_filename(filename):
    """
    Extract angle information from filename, supporting different naming formats.
    
    Description:
        This function handles different filename patterns for angle extraction,
        such as 'deg000', 'deg018_1', etc., ensuring consistent angle value extraction.
    
    Args:
        filename (str): The filename to extract angle from
    
    Returns:
        int or None: The extracted angle value as an integer, or None if not found
        
    References:
        - Regular expressions: https://docs.python.org/3/library/re.html
    """
    # Get the base filename without extension
    base_filename = os.path.basename(filename)
    
    # Pattern 1: Extract any digits after 'deg', handling variations like:
    # - deg000
    # - deg018_1
    # - deg036_1_1000hz
    pattern = re.compile(r'deg(\d+)')
    match = pattern.search(base_filename)
    
    if match:
        try:
            angle = int(match.group(1))
            return angle
        except ValueError:
            pass
    
    return None

def extract_metadata_from_filename(filename):
    """
    Extract comprehensive metadata from filename including material, frequency, index, etc.
    
    Description:
        Parses the filename to extract metadata such as material type, frequency,
        sequence index, and angle information based on common naming patterns.
    
    Args:
        filename (str): The filename to extract metadata from
    
    Returns:
        dict: Dictionary containing extracted metadata fields
        
    References:
        - Regular expressions: https://docs.python.org/3/library/re.html
    """
    # Get the base filename without extension and path
    base_filename = os.path.basename(filename)
    name_without_ext = os.path.splitext(base_filename)[0]
    parts = name_without_ext.split('_')
    
    metadata = {
        'filename': base_filename,
        'full_path': filename
    }
    
    try:
        # Material (usually the first part)
        if parts and len(parts) > 0:
            metadata['material'] = parts[0]
        
        # Frequency (part containing 'hz')
        freq_part = next((p for p in parts if 'hz' in p.lower()), None)
        if freq_part:
            try:
                metadata['frequency'] = int(re.sub(r'[^0-9]', '', freq_part))
            except ValueError:
                pass
        
        # Index - handle both explicit index parts and numeric parts after deg
        # This handles patterns like:
        # - box_sinewave_deg018_1_500hz_00.wav (where 1 is the index)
        # - box_sinewave_deg000_500hz_03.wav (where no explicit index exists)
        
        # Check for single digit after deg pattern
        index_after_deg = False
        for i, part in enumerate(parts):
            if 'deg' in part.lower() and i+1 < len(parts):
                next_part = parts[i+1]
                if len(next_part) == 1 and next_part.isdigit():
                    metadata['index'] = int(next_part)
                    index_after_deg = True
                    break
        
        # If no index found after deg, look for standalone digits
        if not index_after_deg:
            for part in parts:
                if part.isdigit() and len(part) == 1:
                    metadata['index'] = int(part)
                    break
        
        # Extract the angle
        angle = extract_angle_from_filename(filename)
        if angle is not None:
            metadata['angle'] = angle
    
    except Exception as e:
        print(f"Warning: Error extracting metadata from {filename}: {e}")
    
    return metadata

def filter_files_by_metadata(files, filter_criteria):
    """
    Filter a list of files based on metadata criteria.
    
    Description:
        Extracts metadata from each file and filters based on provided criteria
        such as material, frequency, index, and angle values.
    
    Args:
        files (list): List of file paths to filter
        filter_criteria (dict): Criteria for filtering with keys:
            - material_filter (str): Material to filter by, or "all"
            - frequency_filter (list): List of frequencies to include
            - index_filter (list): List of indices to include
            - angle_values (list): List of angle values to include
    
    Returns:
        tuple: (filtered_files, filtered_labels, stats)
            - filtered_files (list): Files that passed the filter
            - filtered_labels (list): Labels corresponding to filtered files
            - stats (dict): Statistics about filtered files
    """
    filtered_files = []
    filtered_labels = []
    
    # Extract filter criteria
    material_filter = filter_criteria.get('material_filter', 'all')
    frequency_filter = filter_criteria.get('frequency_filter', None)
    index_filter = filter_criteria.get('index_filter', None)
    angle_values = filter_criteria.get('angle_values', None)
    
    # Ensure filters are lists
    if frequency_filter and not isinstance(frequency_filter, list):
        frequency_filter = [frequency_filter]
    if index_filter and not isinstance(index_filter, list):
        index_filter = [index_filter]
    
    # Statistics for filtered files
    stats = {
        'total': len(files),
        'accepted': 0,
        'rejected': {
            'material': 0,
            'frequency': 0,
            'index': 0,
            'angle': 0,
            'missing_data': 0,
            'parsing_error': 0
        }
    }
    
    for file in files:
        try:
            metadata = extract_metadata_from_filename(file)
            
            # Apply filters
            
            # Material filter
            if material_filter != 'all' and metadata.get('material') != material_filter:
                stats['rejected']['material'] += 1
                continue
                
            # Frequency filter
            if frequency_filter and metadata.get('frequency') not in frequency_filter:
                stats['rejected']['frequency'] += 1
                continue
                
            # Index filter
            if index_filter and metadata.get('index') not in index_filter:
                stats['rejected']['index'] += 1
                continue
                
            # Angle filter
            if angle_values:
                if 'angle' not in metadata:
                    stats['rejected']['missing_data'] += 1
                    continue
                    
                angle = metadata['angle']
                if angle not in angle_values:
                    stats['rejected']['angle'] += 1
                    continue
                    
                # Get the index of the angle in angle_values for label
                angle_index = angle_values.index(angle)
                filtered_labels.append(angle_index)
                
            filtered_files.append(file)
            stats['accepted'] += 1
            
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            stats['rejected']['parsing_error'] += 1
    
    return filtered_files, filtered_labels, stats 