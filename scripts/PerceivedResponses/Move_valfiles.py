import os
import shutil

# Source and destination directories
src_root = "D:\\shared_git\\MaestriaThesis\\NeuroSenseDatabase"
dest_root = "D:\\shared_git\\MaestriaThesis\\ValidationPreprocessed"

# Subfolders range
subfolders = [f"sub-{i:02d}" for i in range(1, 35)]

for subfolder in subfolders:
    # Source path
    src_path = os.path.join(src_root, subfolder, "pp_validation")

    # File names to copy
    file_names = [f"{subfolder}_B_pp_validation.set", f"{subfolder}_E_pp_validation.set"]

    for file_name in file_names:
        src_file_path = os.path.join(src_path, file_name)
        dest1_file_path = os.path.join(dest_root, subfolder)
        dest2_file_path = os.path.join(dest_root, subfolder, 'pp_validation')

        if not os.path.exists(dest1_file_path):
            os.mkdir(dest1_file_path)
        if not os.path.exists(dest2_file_path):
            os.mkdir(dest2_file_path)      
    
        dest_file_path = os.path.join(dest_root, subfolder, 'pp_validation', file_name)
        # Copy file if it exists
        if os.path.exists(src_file_path):
            shutil.copy(src_file_path, dest_file_path)
            print(f"Copied {file_name} to {dest_root}")
        else:
            print(f"File {file_name} not found in {src_path}")