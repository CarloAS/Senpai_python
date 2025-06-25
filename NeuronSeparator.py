import numpy as np
import scipy.ndimage
import skimage.segmentation
# import skimage.measure # Not strictly needed if using scipy.ndimage.label and .sum_labels
from skimage.io import imread # Added for TIFF reading
from skimage import measure # Explicitly kept for marching_cubes in visualization
import json # No longer needed for soma loading by NeuronSeparator, but kept if other parts of a larger project use it
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for projection='3d'

class NeuronSeparator:
    """
    Class to separate individual neurons from a binary segmentation
    based on an original image and a soma binary mask TIFF image.
    This class translates and adapts functionality similar to the MATLAB script 'senpai_separator.m'.
    """

    def __init__(self):
        """
        Initializes the NeuronSeparator class.
        """
        pass

    def _load_soma_mask_from_tiff(self, soma_mask_path: str, image_shape: tuple) -> np.ndarray:
        """
        Loads a soma binary mask from a TIFF file.

        Args:
            soma_mask_path (str): Path to the TIFF file containing the soma binary mask.
                                  The TIFF should have 0s for background and non-zero
                                  values (typically 1s) for soma locations.
            image_shape (tuple): The shape of the image (Nx, Ny, Nz) for validation.

        Returns:
            np.ndarray: A boolean numpy array with the same shape as the image,
                        where soma locations are marked True. Returns None on error.
        """
        print(f"Loading soma mask from TIFF: {soma_mask_path}")
        path = Path(soma_mask_path)
        if not path.is_file():
            print(f"Error: Soma mask TIFF file not found: {soma_mask_path}")
            return None

        try:
            somas_from_file = imread(soma_mask_path)
        except Exception as e:
            print(f"Error reading soma mask TIFF file {soma_mask_path}: {e}")
            return None

        if somas_from_file.ndim != 3:
            print(f"Error: Soma mask from TIFF must be a 3D array, but got {somas_from_file.ndim} dimensions.")
            return None
        somas_from_file = somas_from_file.transpose((1, 2, 0)) # Transpose to match (Nx, Ny, Nz) if needed
        if somas_from_file.shape != image_shape:
            print(f"Error: Shape of soma mask {somas_from_file.shape} "
                  f"does not match original image shape {image_shape}.")
            return None

        # Ensure the soma mask is binary (boolean)
        # Values > 0 will become True, 0 will become False.
        somas_from_file.transpose((1, 2, 0)) # Transpose to match (Nx, Ny, Nz) if needed
        somas_binary_mask = somas_from_file.astype(bool)

        if not np.any(somas_binary_mask):
            print("Warning: The loaded soma mask is empty (all zeros). Further processing might not yield results.")
            # Allow proceeding as watershed with no markers might still be desired by some,
            # but the standard logic expects markers. The subsequent check for num_soma_labels will handle this

        return somas_binary_mask

    def separate_neurons(self, binary_segmentation: np.ndarray,
                         original_image: np.ndarray,
                         soma_mask_path: str) -> np.ndarray:
        """
        Separates neurons using watershed guided by soma locations from a TIFF mask.

        Args:
            binary_segmentation (np.ndarray): Boolean array of the overall neuron segmentation.
                                              Shape (Nx, Ny, Nz).
            original_image (np.ndarray): The original intensity image. Shape (Nx, Ny, Nz).
            soma_mask_path (str): Path to the TIFF file containing the soma binary mask.
                                   The mask should have the same dimensions as the original_image.

        Returns:
            np.ndarray: Label matrix (dtype=np.uint16) where each unique integer
                        corresponds to a separated neuron. Shape (Nx, Ny, Nz).
                        Returns None if critical errors occur (e.g., invalid soma mask file).
        """
        print('Separating neurons...')

        if not (binary_segmentation.ndim == 3 and original_image.ndim == 3):
            raise ValueError("Inputs binary_segmentation and original_image must be 3D arrays.")
        if binary_segmentation.shape != original_image.shape:
            raise ValueError("Shapes of binary_segmentation and original_image must match.")

        senpai_seg = binary_segmentation.astype(bool)

        somas_binary_mask = self._load_soma_mask_from_tiff(soma_mask_path, original_image.shape)

        if somas_binary_mask is None: # Error during loading
            print("Error: Failed to load or validate soma mask. Cannot proceed.")
            return None

        if not np.any(somas_binary_mask):
            print("Error: Soma mask is empty (all zeros after loading). Cannot use for guided watershed.")
            return None

        # Label the soma mask to serve as markers for watershed
        # Each connected component in somas_binary_mask will be a unique marker.
        soma_conn_structure = scipy.ndimage.generate_binary_structure(3, 1) # 6-connectivity for labeling somas
        labeled_somas, num_soma_labels = scipy.ndimage.label(somas_binary_mask, structure=soma_conn_structure)

        if num_soma_labels == 0:
             print("Error: Somas mask is empty after labeling (no distinct soma objects found). Cannot proceed.")
             return None
        print(f"Found {num_soma_labels} distinct soma markers.")

        print('Filtering image and preparing for watershed...')
        # Median filter is applied to the original image, similar to medfilt3 in MATLAB
        filtered_image = scipy.ndimage.median_filter(original_image, size=3)

        # Prepare image for watershed: take the negative of the filtered image
        # This makes regions of high intensity (neurons) into basins
        image_max_intensity = np.max(original_image) # Similar to db=max(cIM(:))
        image_for_watershed = image_max_intensity - filtered_image

        # Apply the overall neuron segmentation (senpai_seg) as a mask for watershed.
        # This means watershed will only operate within the segmented regions.
        # In MATLAB: cIM_inv(~senpai_seg)=db; (setting areas outside segmentation to max intensity,
        # effectively preventing watershed from flowing there).
        # skimage.segmentation.watershed's 'mask' parameter achieves this.

        # The imimposemin equivalent is handled by using `labeled_somas` as markers in watershed.

        watershed_conn = scipy.ndimage.generate_binary_structure(3, 1) # Connectivity for watershed expansion
        print('Performing watershed...')
        watershed_result = skimage.segmentation.watershed(
            image_for_watershed,
            markers=labeled_somas,
            mask=senpai_seg,
            connectivity=watershed_conn, # Defines neighborhood for watershed algorithm
            watershed_line=False # Keeps watershed lines as part of labeled regions if True; False merges them.
                                 # For neuron separation, False is usually desired.
        )
        parcel_final = watershed_result.astype(np.uint16) # Cast to uint16 like in MATLAB

        print('Pruning non-connected branches...')
        unique_labels = np.unique(parcel_final)
        unique_labels = unique_labels[unique_labels != 0] # Exclude background

        # Connectivity for pruning, 6-connectivity in 3D
        pruning_conn_structure = scipy.ndimage.generate_binary_structure(3, 1)

        for label_val in unique_labels:
            current_object_mask = (parcel_final == label_val)
            # Find all connected components within the mask for the current neuron label
            labeled_components, num_components = scipy.ndimage.label(current_object_mask, structure=pruning_conn_structure)

            if num_components > 1: # If there's more than one disconnected piece
                # Calculate the size of each component
                component_sizes = scipy.ndimage.sum_labels(
                    np.ones(current_object_mask.shape, dtype=float), # Count pixels in each component
                    labeled_components,
                    index=np.arange(1, num_components + 1) # Labels are 1-indexed
                )

                if not component_sizes.size: # Should not happen if num_components > 1
                    continue

                # Find the label of the largest component
                # np.argmax returns the index in component_sizes, add 1 because labels are 1-indexed
                largest_component_label_in_hist = np.argmax(component_sizes) + 1

                # Create a mask for all pixels that are part of smaller (non-largest) components
                pixels_to_remove_mask = (labeled_components != largest_component_label_in_hist) & (labeled_components != 0)
                parcel_final[pixels_to_remove_mask] = 0 # Set smaller components to background
        
        print('Done!')
        return parcel_final


def visualize_parcellated_neurons_3d(
        parcellated_array: np.ndarray,
        max_neurons: int = 15,
        alpha: float = 0.7,
        view_angle: tuple = (30, 30),
        colormap_name: str = 'tab20',
        title: str = '3D Visualization of Parcellated Neurons'):
    """
    Creates a 3D visualization of a parcellated neuron array, where each
    unique ID is a different neuron, rendered with a distinct color.

    Args:
        parcellated_array (np.ndarray): 3D numpy array where each integer value
                                        represents a unique neuron ID (0 is background).
        fig_size (tuple): Figure size (width, height) in inches.
        alpha (float): Transparency of the surfaces (0-1).
        view_angle (tuple): Initial viewing angle (elevation, azimuth) in degrees.
        colormap_name (str): Name of the matplotlib colormap to use for coloring neurons
                             (e.g., 'tab10', 'tab20', 'viridis', 'jet').
        title (str): The title for the plot.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    """if not isinstance(parcellated_array, np.ndarray) or parcellated_array.ndim != 3:
        raise ValueError("parcellated_array must be a 3D NumPy array.")"""

    unique_labels = np.unique(parcellated_array)
    neuron_labels = unique_labels[unique_labels != 0] # Exclude background (label 0)
    fig_size = (12, 10),
    if neuron_labels.size == 0:
        print("No neurons found in the parcellated array (only background label 0).")
        # Create an empty plot or just return
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("No neurons to display")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        return fig

    # Set up colormap
    try:
        colormap = plt.cm.get_cmap(colormap_name)
    except ValueError:
        print(f"Warning: Colormap '{colormap_name}' not found. Defaulting to 'tab20'.")
        colormap = plt.cm.get_cmap('tab20')
    
    # Ensure enough distinct colors if the colormap has a fixed number (like tab10/tab20)
    if hasattr(colormap, 'N') and colormap.N < len(neuron_labels):
        print(f"Warning: The chosen colormap '{colormap_name}' has {colormap.N} distinct colors, "
              f"but there are {len(neuron_labels)} neurons. Colors will repeat.")
        # For continuous colormaps, we'll map normalized label values to colors.
        # For qualitative (listed) colormaps, repetition is handled by modulo.
        colors = [colormap(i % colormap.N) for i in range(len(neuron_labels))]
    elif not hasattr(colormap, 'N'): # Continuous colormap
        colors = [colormap(i / float(len(neuron_labels) -1 if len(neuron_labels) > 1 else 1 )) for i in range(len(neuron_labels))]
    else: # Qualitative colormap with enough colors
        colors = [colormap(i) for i in range(len(neuron_labels))]


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print(f"Visualizing {len(neuron_labels)} neurons...")
    all_verts_x = []
    all_verts_y = []
    all_verts_z = []

    for i, label_id in enumerate(neuron_labels[:15]):
        neuron_mask = (parcellated_array == label_id)
        
        # Use marching cubes to find the surface mesh.
        # The level should be 0.5 for a binary mask (0s and 1s).
        try:
            verts, faces, normals, values = measure.marching_cubes(
                neuron_mask.astype(np.uint8), level=0.5
            )
            
            # Collect all vertices for bounding box calculation
            if verts.size > 0:
                all_verts_x.extend(verts[:, 0])
                all_verts_y.extend(verts[:, 1])
                all_verts_z.extend(verts[:, 2])

                # Plot the surface for this neuron
                mesh = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                                       color=colors[i % len(colors)], # Modulo for safety if colors list is shorter
                                       alpha=alpha,
                                       edgecolor='none', # No wireframe for cleaner look
                                       linewidth=0, antialiased=True) # Ensure antialiased is True for smooth edges
        except Exception as e:
            print(f"Could not generate mesh for neuron ID {label_id}: {e}")
            continue
    
    if not all_verts_x: # No meshes were generated
        print("No renderable surfaces found for any neuron.")
        ax.set_title("No renderable neuron surfaces")
    else:
        # Set plot limits and aspect ratio after all objects are plotted
        # Manual aspect ratio setting for a more 'cubic' view
        min_x, max_x = min(all_verts_x), max(all_verts_x)
        min_y, max_y = min(all_verts_y), max(all_verts_y)
        min_z, max_z = min(all_verts_z), max(all_verts_z)

        x_range = max_x - min_x
        y_range = max_y - min_y
        z_range = max_z - min_z
        
        plot_max_range = max(x_range, y_range, z_range) / 2.0
        if plot_max_range == 0: # Handle case of single point or flat object
            plot_max_range = 1 

        mid_x = (max_x + min_x) / 2.0
        mid_y = (max_y + min_y) / 2.0
        mid_z = (max_z + min_z) / 2.0

        ax.set_xlim(mid_x - plot_max_range, mid_x + plot_max_range)
        ax.set_ylim(mid_y - plot_max_range, mid_y + plot_max_range)
        ax.set_zlim(mid_z - plot_max_range, mid_z + plot_max_range)
        ax.set_title(title)

    # Set labels and view angle
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    plt.tight_layout()
    plt.show()

    return fig