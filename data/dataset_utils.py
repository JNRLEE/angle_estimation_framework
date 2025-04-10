import re
import os

def extract_angle_from_path(file_path):
    """
    Extract angle information from file path in the new folder structure.
    
    Description:
        Extracts angle value from the directory structure where files are organized
        in angle-specific folders (e.g., deg000, deg018).
    
    Args:
        file_path (str): The full path to the audio file
    
    Returns:
        int or None: The extracted angle value as an integer, or None if not found
        
    References:
        - Regular expressions: https://docs.python.org/3/library/re.html
    """
    # Extract angle from path (e.g., .../deg018/box/file.wav -> 18)
    path_parts = file_path.split(os.sep)
    for part in path_parts:
        if part.startswith('deg'):
            try:
                # Extract the angle value from the folder name
                angle_match = re.search(r'deg(\d+)', part)
                if angle_match:
                    return int(angle_match.group(1))
            except ValueError:
                pass
    
    # If angle not found in path, try extracting from filename as fallback
    return extract_angle_from_filename(file_path)

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

def extract_material_from_path(file_path):
    """
    Extract material information from file path in the new folder structure.
    
    Description:
        Extracts material type (box, plastic) from the directory structure.
    
    Args:
        file_path (str): The full path to the audio file
    
    Returns:
        str or None: The extracted material type, or None if not found
    """
    # Extract material from path (e.g., .../deg018/box/file.wav -> box)
    path_parts = file_path.split(os.sep)
    materials = ['box', 'plastic']
    
    for part in path_parts:
        if part.lower() in materials:
            return part.lower()
    
    # If material not found in path, try extracting from filename
    base_filename = os.path.basename(file_path)
    name_parts = os.path.splitext(base_filename)[0].split('_')
    
    if name_parts and name_parts[0].lower() in materials:
        return name_parts[0].lower()
    
    return None

def extract_metadata_from_filename(filename):
    """
    Extract comprehensive metadata from filename and path including material, frequency, index, etc.
    
    Description:
        Parses the file path and filename to extract metadata such as material type, frequency,
        sequence index, and angle information based on the new directory structure.
    
    Args:
        filename (str): The full path to the audio file
    
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
        # Extract angle from path
        angle = extract_angle_from_path(filename)
        if angle is not None:
            metadata['angle'] = angle
        
        # Extract material from path
        material = extract_material_from_path(filename)
        if material:
            metadata['material'] = material
        elif parts and len(parts) > 0:
            # Fallback to extracting from filename
            metadata['material'] = parts[0]
        
        # Frequency (part containing 'hz')
        freq_part = next((p for p in parts if 'hz' in p.lower()), None)
        if freq_part:
            try:
                metadata['frequency'] = int(re.sub(r'[^0-9]', '', freq_part))
            except ValueError:
                pass
        
        # Index - usually the last part (after frequency)
        for part in reversed(parts):
            if part.isdigit() and len(part) <= 2:  # Typically index is 1-2 digits
                metadata['index'] = int(part)
                break
    
    except Exception as e:
        print(f"Warning: Error extracting metadata from {filename}: {e}")
    
    return metadata

def get_files_from_hierarchical_structure(root_dir, format_filter=None):
    """
    Recursively find all files in the hierarchical structure.
    
    Description:
        Traverses the hierarchical directory structure to find all audio files,
        organized as root_dir/deg{angle}/{material}/{files}.
    
    Args:
        root_dir (str): Root directory containing the hierarchical structure
        format_filter (list, optional): List of file extensions to include (e.g., ['wav'])
    
    Returns:
        list: List of file paths found in the structure
    """
    all_files = []
    
    # Ensure format_filter is a list
    if format_filter and not isinstance(format_filter, list):
        format_filter = [format_filter]
    
    # Walk through the directory structure
    for angle_dir in sorted(os.listdir(root_dir)):
        angle_path = os.path.join(root_dir, angle_dir)
        
        # Skip if not a directory or doesn't match deg pattern
        if not os.path.isdir(angle_path) or not angle_dir.startswith('deg'):
            continue
            
        for material_dir in sorted(os.listdir(angle_path)):
            material_path = os.path.join(angle_path, material_dir)
            
            # Skip if not a directory or not a valid material
            if not os.path.isdir(material_path) or material_dir.lower() not in ['box', 'plastic']:
                continue
                
            # Get all files in this material directory
            for file in os.listdir(material_path):
                file_path = os.path.join(material_path, file)
                
                # Skip if not a file
                if not os.path.isfile(file_path):
                    continue
                    
                # Apply format filter if specified
                if format_filter:
                    ext = os.path.splitext(file)[1].lower().lstrip('.')
                    if ext not in format_filter:
                        continue
                        
                all_files.append(file_path)
    
    return all_files

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