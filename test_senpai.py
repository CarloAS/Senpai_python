import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
import pickle
from SENPAI import (
    SegmentationConfig,
    NeuralSegmentation,
    NeuronSeparator,
    NeuronSkeletonizer,
    NeuronPruner
)
from skimage.measure import *
from skimage.morphology import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Get current working directory
current_fold = os.getcwd()

# Define configuration parameters
# Note: These parameters match the MATLAB version's defaults
config = SegmentationConfig(
    sig_G=[1.0, 1.0, 1.0],  # Gaussian smoothing parameters
    size_lim=[256, 256, 149],  # Size limits matching MATLAB script
    clusters=6,  # K=6 as defined in MATLAB script
    verbmem=False,
    paralpool=True
)

# Initialize segmentation
segmentation = NeuralSegmentation(config)

# Create results directory
results_dir = os.path.join(current_fold, 'res_test_full')
os.makedirs(results_dir, exist_ok=True)

# Load and segment image
segmentation.load_image(
    path=current_fold,
    filename='test_data.tif',
    output_path=results_dir
)
results = segmentation.segment_image()

# Load soma markers and additional markers
# Note: These would need to be created/loaded from compatible numpy arrays
try:
    #somas = np.load('somas.npy')
    """with open('markers.pkl', 'rb') as f:
        markers = pickle.load(f)"""
    
    somas = loadmat('somas.mat')
    markers = loadmat('markers.mat')
    
except FileNotFoundError:
    print("Soma or marker files not found. Creating dummy data for testing...")
    somas = np.zeros_like(results['segmentation'], dtype=bool)
    markers = np.zeros_like(results['segmentation'], dtype=bool)

# Initialize and run neuron separation
separator = NeuronSeparator(
    segmentation=results['segmentation'],
    original_image=results['original']
)
parcellation = separator.separate_neurons(somas | markers)

# Visualize selected neurons (equivalent to MATLAB visualization)
selected_neurons = [73, 30, 43, 12, 13, 11, 16, 10, 8, 6]

# Create directory for neuron morphology
morph_dir = Path('neurons_morph')
morph_dir.mkdir(exist_ok=True)

# Setup 3D visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Process each selected neuron
for neuron_id in selected_neurons:
    # Create mask for current neuron
    mask = parcellation == neuron_id
    
    if np.any(mask):
        # Create skeletonizer for current neuron
        skeletonizer = NeuronSkeletonizer(
            image=results['original'],
            neuron_mask=mask
        )
        
        # Generate skeleton and save to file
        graph, swc = skeletonizer.create_skeleton(somas)
        np.savetxt(
            morph_dir / f'neuron_{neuron_id}_skeleton.swc',
            swc,
            fmt='%d %d %.6f %.6f %.6f %.6f %d'
        )
        
        # Visualize neuron surface
        verts, faces, _, _ = marching_cubes(mask, 0.2)
        color = np.random.rand(3)
        mesh = Poly3DCollection(verts[faces], alpha=0.8)
        mesh.set_facecolor(color)
        ax.add_collection3d(mesh)

# Set visualization properties
ax.set_title('Selection of close-by neurons')
ax.view_init(elev=-20, azim=60)
ax.axis('off')
plt.tight_layout()

# Save figure
plt.savefig(morph_dir / 'selected_neurons.png')

# Initialize and run pruning interface if needed
"""pruner = NeuronPruner(parcellation)
pruner.run()"""
