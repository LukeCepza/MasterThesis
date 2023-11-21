import numpy as np
import scipy.io
import os
from nilearn import plotting
import matplotlib.pyplot as plt

def tal2mni(inpoints):
    s1, s2 = inpoints.shape
    if s1 != 3:
        raise ValueError('Input must be a 3xN matrix')

    # Transformation matrices, different zooms above/below AC
    M2T = mni2tal_matrix()

    inpoints = np.vstack((inpoints, np.ones((1, s2))))

    tmp = inpoints[2, :] < 0  # 1 if below AC

    inpoints[:, tmp] = np.linalg.solve((M2T['rotn'] @ M2T['downZ']).T, inpoints[:, tmp])
    inpoints[:, ~tmp] = np.linalg.solve((M2T['rotn'] @ M2T['upZ']).T, inpoints[:, ~tmp])

    outpoints = inpoints[:3, :]
    
    return outpoints

def mni2tal_matrix():
    M2T = {}

    # rotn  = spm_matrix([0 0 0 0.05]); % similar to Rx(eye(3),-0.05), DLW
    M2T['rotn'] = np.array([
        [1, 0, 0, 0],
        [0, 0.9988, 0.0500, 0],
        [0, -0.0500, 0.9988, 0],
        [0, 0, 0, 1]
    ])

    # upz   = spm_matrix([0 0 0 0 0 0 0.99 0.97 0.92]);
    M2T['upZ'] = np.array([
        [0.9900, 0, 0, 0],
        [0, 0.9700, 0, 0],
        [0, 0, 0.9200, 0],
        [0, 0, 0, 1]
    ])

    # downz = spm_matrix([0 0 0 0 0 0 0.99 0.97 0.84]);
    M2T['downZ'] = np.array([
        [0.9900, 0, 0, 0],
        [0, 0.9700, 0, 0],
        [0, 0, 0.8400, 0],
        [0, 0, 0, 1]
    ])

    return M2T


# Create a 3x4 subplot grid
fig, axes = plt.subplots(3, 5, figsize=(20, 12))

# Specify the directory path
directory_path = r'D:\shared_git\MaestriaThesis\results\EEGLAB_STUDY'

# List all files in the directory
file_list = os.listdir(directory_path)

# Filter for .mat files
mat_files = [file for file in file_list if file.endswith('.mat')]
a = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] #np.arange(0,15)
# Loop through each .mat file
for idx, mat_file in enumerate(mat_files):
    mat_file_path = os.path.join(directory_path, mat_file)
    # Load the MATLAB .mat file
    mat_data = scipy.io.loadmat(mat_file_path)
    locs_data = mat_data['locs'][1:, :]  # Exclude the first row
    dmn_coords = locs_data #[(0, -52, 18), (-46, -68, 32), (46, -68, 32), (1, 50, -5)]
    # Plot in the current subplot
    ax = axes[idx // 5, idx % 5]
    # Set smaller and centered title without '_loc.mat' part
    title = mat_file.replace('_locs.mat', '')
    plotting.plot_markers(a,dmn_coords, node_size = 20, node_cmap = 'brg', axes = ax, display_mode = 'z', colorbar = False)
    ax.set_title(title, fontsize=8, ha='center')

# Adjust layout for better spacing
plt.tight_layout()
plt.show()
