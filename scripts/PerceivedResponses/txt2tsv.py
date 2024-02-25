import os
import shutil

# Source and destination directories
src_root = "D:\\shared_git\\MaestriaThesis\\NeuroSenseDatabase"

# Subfolders range
subfolders = [f"sub-{i:02d}" for i in range(1, 35)]

for subfolder in subfolders:
    
    file_names = [f"{subfolder}_R.txt", f"{subfolder}_R.tsv"]

    # Source path
    src = os.path.join(src_root, subfolder, "other",file_names[0])
    output = os.path.join(src_root, subfolder, "other",file_names[1])

    with open(src, 'r') as infile, open(output, 'w') as outfile:
        for line in infile:
            # Split the line into individual digits and join with a space
            formatted_line = '\t'.join(list(line.strip()))
            # Write the formatted line to the .tsv file
            outfile.write(formatted_line + '\n')
        # File names to copy
            