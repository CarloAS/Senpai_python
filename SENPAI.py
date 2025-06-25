import os
import tifffile
import numpy as np
import scipy.io as sio
import warnings
import scipy.ndimage as ndi
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from skimage.filters import gaussian
from skimage import io
from skimage.segmentation import watershed
from skimage.measure import *
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage.morphology import *
from matplotlib.widgets import Button, TextBox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class SegmentationConfig:
    """Configuration parameters for image segmentation."""
    sig_G: List[float]  # Gaussian smoothing parameters
    size_lim: List[int]  # Size limits for processing
    clusters: int       # Number of clusters for k-means
    verbmem: bool = False  # Whether to save intermediate results
    paralpool: bool = True  # Whether to use parallel processing

class ImageMetadata:
    """Class to handle image metadata and dimensions."""
    def __init__(self, path: str, filename: str):
        print(f"\nInitializing ImageMetadata for {filename}")
        """
        Initialize image metadata from file.
        
        Args:
            path: Directory containing the image
            filename: Name of the image file
        """
        self.path = path
        self.filename = filename
        self._load_metadata()

    def _load_metadata(self):
        """Load and parse metadata from TIFF file."""
        with tifffile.TiffFile(os.path.join(self.path, self.filename)) as tif:
            metadata = tif.pages[0].tags
            self.Ny, self.Nx = tif.pages[0].shape
            self.Nz = len(tif.pages)
            self.bit_depth = metadata['BitsPerSample'].value
            self.xres = metadata.get('XResolution', 1)
            self.yres = metadata.get('YResolution', 1)
            self.zres = self._extract_z_resolution(metadata)
            self.dtype = np.float32 if self.bit_depth == 16 else np.uint8

    def _extract_z_resolution(self, metadata) -> float:
        """
        Extract z-resolution from metadata.
        
        Args:
            metadata: TIFF metadata
            
        Returns:
            float: Z-resolution value
        """
        description = metadata.get('ImageDescription', None)
        if description:
            description = description.value
            if 'PhysicalSizeZ=' in description:
                return float(description.split('PhysicalSizeZ=')[1].split()[0])
            elif 'spacing=' in description:
                return float(description.split('spacing=')[1].split()[0])
        return 1.0

class NeuralSegmentation:
    """Main class for neural image segmentation and analysis."""
    def __init__(self, config: SegmentationConfig):
        print("\nInitializing Neural Segmentation")
        """
        Initialize segmentation with configuration.
        
        Args:
            config: SegmentationConfig instance with parameters
        """
        self.config = config
        self.metadata = None
        self.output_path = None
        self.segmentation_result = None

    def load_image(self, path: str, filename: str, output_path: Optional[str] = None):
        """
        Load image and setup output directory.
        
        Args:
            path: Directory containing the image
            filename: Name of the image file
            output_path: Directory for saving results (optional)
        """
        self.metadata = ImageMetadata(path, filename)
        self.output_path = output_path or os.path.join(os.getcwd(), 'senpaiseg')
        os.makedirs(self.output_path, exist_ok=True)

    def _load_image_data(self) -> np.ndarray:
        """
        Load image data from file.
        
        Returns:
            np.ndarray: Loaded image data
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return io.imread(os.path.join(self.metadata.path, self.metadata.filename))
        

    def _perform_kmeans_clustering(self, image_data: np.ndarray) -> np.ndarray:
        print("\nPerforming k-means clustering...")
        """
        Perform k-means clustering on image data.
        
        Args:
            image_data: Input image data
            
        Returns:
            np.ndarray: Clustered image data
        """
        def _resolve_resolution(resolution):
            """Resolve TIFF resolution metadata."""
            if isinstance(resolution, tuple):
                return resolution[0] / resolution[1]
            return float(resolution)
        
        th_back = 0.02 * (2 ** self.metadata.bit_depth)
        
        if self.config.sig_G[0] > 0:
            smoothed = gaussian(image_data, sigma=(
                self.config.sig_G[0],
                self.config.sig_G[0] * _resolve_resolution(self.metadata.xres.value) / _resolve_resolution(self.metadata.yres.value),
                self.config.sig_G[0] * _resolve_resolution(self.metadata.xres.value) / float(self.metadata.zres)
            ))
            Gx, Gy, Gz = np.gradient(smoothed)
        else:
            Gx, Gy, Gz = np.gradient(image_data)
        
        Gxx = np.gradient(Gx, axis=0)
        Gyy = np.gradient(Gy, axis=1)
        Gzz = np.gradient(Gz, axis=2)
        
        mask_back = image_data >= th_back
        km_in = np.stack([
            image_data[mask_back],
            Gxx[mask_back],
            Gyy[mask_back],
            Gzz[mask_back]
        ], axis=1)
        
        if km_in.shape[0] >= 10:
            kmeans = KMeans(
                n_clusters=self.config.clusters,
                n_init=10,
                max_iter=1000,
                random_state=42
            )
            clusters = kmeans.fit_predict(km_in)
            output = np.zeros_like(image_data, dtype=np.uint8)
            output[mask_back] = clusters + 1
        else:
            output = np.ones_like(image_data, dtype=np.uint8)
        
        return output

    def _recompose_segments(self, segmented_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Recompose segmented data into final form.
        
        Args:
            segmented_data: Clustered image data
            
        Returns:
            Dict containing final segmentation results
        """
        # Initialize output matrices
        final_seg = np.zeros_like(segmented_data, dtype=np.uint8)
        
        # Process each slice
        for z in range(segmented_data.shape[2]):
            final_seg[:,:,z] = segmented_data[:,:,z]
        
        # Create result dictionary
        result = {
            'segmentation': final_seg,
            'original': self._load_image_data()
        }
        
        # Save results
        save_path = os.path.join(self.output_path, 'segmentation_result.npz')
        np.savez_compressed(save_path, **result)
        
        return result

    def segment_image(self) -> Dict[str, np.ndarray]:
        print("\nStarting image segmentation...")
        """
        Perform complete image segmentation.
        
        Returns:
            Dict containing segmentation results
        """
        print("Loading image data...")
        image_data = self._load_image_data()
        print("Performing segmentation...")
        segmented = self._perform_kmeans_clustering(image_data)
        print("Recomposing segments...")
        self.segmentation_result = self._recompose_segments(segmented)
        print("Segmentation completed")
        return self.segmentation_result

class NeuronSeparator:
    """Class for separating individual neurons from segmentation."""
    def __init__(self, segmentation: np.ndarray, original_image: np.ndarray):
        print("\nInitializing Neuron Separator")
        """
        Initialize separator with image data.
        
        Args:
            segmentation: Binary segmentation mask
            original_image: Original image data
        """
        self.segmentation = segmentation
        self.original_image = original_image

    def separate_neurons(self, soma_mask: np.ndarray) -> np.ndarray:
        print("\nStarting neuron separation")
        """
        Separate neurons using watershed transformation.
        
        Args:
            soma_mask: Binary mask of soma locations
            
        Returns:
            np.ndarray: Separated neuron labels
        """
        print("Inverting image...")
        inverted_image = self._invert_image()
        print("Applying watershed transformation...")
        watershed_result = self._apply_watershed(inverted_image, soma_mask)
        print("Pruning disconnected branches...")
        return self._prune_disconnected_branches(watershed_result)

    def _invert_image(self) -> np.ndarray:
        """
        Invert image for watershed transform.
        
        Returns:
            np.ndarray: Inverted image
        """
        db = np.max(self.original_image)
        inv = db - ndi.median_filter(self.original_image, size=3)
        inv[~self.segmentation] = db
        return inv

    def _apply_watershed(self, inv_image: np.ndarray, soma_mask: np.ndarray) -> np.ndarray:
        """
        Apply watershed transformation.
        
        Args:
            inv_image: Inverted image
            soma_mask: Binary soma mask
            
        Returns:
            np.ndarray: Watershed result
        """
        inv_image = np.where(soma_mask, 0, inv_image)
        return watershed(inv_image) * self.segmentation.astype(np.uint16)

    def _prune_disconnected_branches(self, parcellation: np.ndarray) -> np.ndarray:
        """
        Remove disconnected components from each neuron.
        
        Args:
            parcellation: Initial watershed parcellation
            
        Returns:
            np.ndarray: Pruned parcellation
        """
        max_label = np.max(parcellation)
        
        for vv in range(1, max_label + 1):
            mask = parcellation == vv
            labeled, num_features = label(mask, connectivity=3, return_num=True)
            
            if num_features > 1:
                regions = regionprops(labeled)
                largest_region = max(regions, key=lambda r: r.area)
                
                for region in regions:
                    if region != largest_region:
                        coords = region.coords
                        parcellation[coords[:, 0], coords[:, 1], coords[:, 2]] = 0
        
        return parcellation

class NeuronSkeletonizer:
    """Class for creating skeletal representations of neurons."""
    def __init__(self, image: np.ndarray, neuron_mask: np.ndarray):
        print("\nInitializing Neuron Skeletonizer")
        """
        Initialize skeletonizer with image data.
        
        Args:
            image: Original image data
            neuron_mask: Binary neuron mask
        """
        self.image = image
        self.neuron_mask = neuron_mask

    def _skeletonize_3d(self, mask: np.ndarray) -> np.ndarray:
        print("\nStarting 3D skeletonization")
        """
        Perform 3D skeletonization using iterative thinning.
        
        Args:
            mask: Binary 3D mask
            
        Returns:
            np.ndarray: Skeletonized mask
        """
        # Initialize the skeleton
        skeleton = np.copy(mask)
        changing = True
        iteration = 0
        max_iterations = 100  # Prevent infinite loops
        
        while changing and iteration < max_iterations:
            # Store previous iteration for comparison
            prev = np.copy(skeleton)
            
            # Perform thinning iteration
            skeleton = self._thin_iteration(skeleton)
            
            # Check if the skeleton has changed
            changing = not np.array_equal(skeleton, prev)
            iteration += 1
        
        return skeleton

    def _thin_iteration(self, mask: np.ndarray) -> np.ndarray:
        """
        Perform one iteration of 3D thinning.
        
        Args:
            mask: Binary 3D mask
            
        Returns:
            np.ndarray: Thinned mask
        """
        # Create output array
        result = np.copy(mask)
        
        # Get all points that are candidates for removal
        points = np.argwhere(mask > 0)
        
        for point in points:
            x, y, z = point
            
            # Get 3x3x3 neighborhood
            neighborhood = mask[max(0, x-1):min(x+2, mask.shape[0]),
                              max(0, y-1):min(y+2, mask.shape[1]),
                              max(0, z-1):min(z+2, mask.shape[2])]
            
            # Check if point can be removed without breaking connectivity
            if self._can_remove_point(neighborhood):
                result[x, y, z] = 0
        
        return result

    def _can_remove_point(self, neighborhood: np.ndarray) -> bool:
        """
        Check if a point can be removed while preserving topology.
        
        Args:
            neighborhood: 3x3x3 binary neighborhood
            
        Returns:
            bool: True if point can be removed
        """
        # Center point must be 1
        if neighborhood[1,1,1] == 0:
            return False
        
        # Must have at least two neighbors to preserve connectivity
        if np.sum(neighborhood) < 3:
            return False
        
        # Check if removing point would create a hole or break connectivity
        # Simple topology preservation test
        neighbors = np.sum(neighborhood) - 1  # Subtract center point
        if neighbors < 2 or neighbors > 6:
            return False
            
        # Must preserve at least one connection in each axis
        axes_connections = [
            neighborhood[0,1,1] + neighborhood[2,1,1],  # x-axis
            neighborhood[1,0,1] + neighborhood[1,2,1],  # y-axis
            neighborhood[1,1,0] + neighborhood[1,1,2]   # z-axis
        ]
        
        if any(conn == 0 for conn in axes_connections):
            return False
            
        return True

    def _generate_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """
        Generate skeleton from mask using custom 3D skeletonization.
        
        Args:
            mask: Binary mask
            
        Returns:
            np.ndarray: Skeletonized mask
        """
        return self._skeletonize_3d(mask)

    def create_skeleton(self, soma_mask: np.ndarray) -> Tuple[nx.Graph, np.ndarray]:
        print("\nCreating neuron skeleton")
        """
        Create skeleton and SWC representation.
        
        Args:
            soma_mask: Binary soma mask
            
        Returns:
            Tuple containing graph and SWC matrix
        """
        processed_mask = self._preprocess_mask(soma_mask)
        skeleton = self._generate_skeleton(processed_mask)
        return self._build_graph(skeleton)

    def _preprocess_mask(self, soma_mask: np.ndarray) -> np.ndarray:
        """
        Preprocess mask for skeletonization.
        
        Args:
            soma_mask: Binary soma mask
            
        Returns:
            np.ndarray: Processed mask
        """
        neuron_mask = self.neuron_mask > 0
        soma_mask = soma_mask > 0
        combined_mask = neuron_mask | soma_mask
        
        labeled_mask, num_features = label(combined_mask, connectivity=3, return_num=True)
        
        if num_features > 1:
            region_sizes = np.array([
                np.sum(labeled_mask == i)
                for i in range(1, num_features + 1)
            ])
            largest_region = np.argmax(region_sizes) + 1
            processed_mask = (labeled_mask == largest_region)
        else:
            processed_mask = combined_mask
        
        processed_mask = np.pad(processed_mask, 1, mode="constant")
        return processed_mask

    def _build_graph(self, skeleton: np.ndarray) -> Tuple[nx.Graph, np.ndarray]:
        """
        Build graph and SWC representation.
        
        Args:
            skeleton: Skeletonized mask
            
        Returns:
            Tuple containing graph and SWC matrix
        """
        points = np.argwhere(skeleton)
        G = nx.Graph()
        
        # Build graph from skeleton points
        for i, point in enumerate(points):
            for j, other_point in enumerate(points[i+1:], i+1):
                dist = np.sum((point - other_point) ** 2)
                if dist <= 3:  # Connect points within sqrt(3) distance
                    weight = self.image[tuple(other_point)]
                    G.add_edge(i, j, weight=weight)
        
        # Create minimum spanning tree
        mst = nx.minimum_spanning_tree(G)
        
        # Create SWC format
        swc = []
        for i, point in enumerate(points):
            radius = ndi.distance_transform_edt(self.neuron_mask)[tuple(point)]
            parent = -1
            edges = list(mst.edges(i))
            if edges:
                parent = edges[0][1]  # Take first connected point as parent
            swc.append([i+1, 0, point[0], point[1], point[2], radius, parent+1])
        
        return mst, np.array(swc)


class NeuronPruner:
    """Interactive GUI for pruning neural branches."""
    
    def __init__(self, parcellation: np.ndarray):
        print("\nInitializing Neuron Pruner")
        """
        Initialize pruning interface.
        Args:
            parcellation: Neuron parcellation
        """
        self.parcellation = parcellation
        self.markers = np.zeros_like(parcellation, dtype=bool)
        self.current_neuron = 1
        self.max_neurons = np.max(parcellation)
        self.fig = None
        self.ax3d = None
        self.scatter = None
        self.selected_points = []
        
    def setup_gui(self):
        """Setup the interactive GUI interface."""
        self.fig = plt.figure(figsize=(10, 8))
        self.ax3d = self.fig.add_subplot(111, projection='3d')
        self.ax3d.axis('off')
        
        # Add GUI controls
        ax_prev = plt.axes([0.35, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.55, 0.05, 0.1, 0.075])
        ax_save = plt.axes([0.75, 0.05, 0.15, 0.075])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_save = Button(ax_save, 'Save')
        
        self.btn_prev.on_clicked(self._prev)
        self.btn_next.on_clicked(self._next)
        self.btn_save.on_clicked(self._save)
        
        # Add click event handling
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        
        # Initial plot
        self._update_plot()
        
    def _update_plot(self):
        """Update the 3D plot with current neuron data."""
        self.ax3d.clear()
        self.ax3d.axis('off')
        
        # Get current neuron coordinates
        mask = self.parcellation == self.current_neuron
        coords = np.where(mask)
        
        if len(coords[0]) > 0:
            # Plot unmarked points
            unmarked = ~self.markers[mask]
            self.scatter = self.ax3d.scatter(
                coords[0][unmarked], 
                coords[1][unmarked], 
                coords[2][unmarked],
                c='b', alpha=0.6
            )
            
            # Plot marked points
            marked = self.markers[mask]
            if np.any(marked):
                self.ax3d.scatter(
                    coords[0][marked],
                    coords[1][marked],
                    coords[2][marked],
                    c='r', alpha=0.6
                )
        
        self.ax3d.set_title(f'Neuron {self.current_neuron}/{self.max_neurons}')
        self.fig.canvas.draw_idle()
    
    def _on_click(self, event):
        """Handle mouse click events for point selection."""
        if event.inaxes != self.ax3d:
            return
            
        if self.scatter is None:
            return
            
        # Get clicked point in data coordinates
        clicked = np.array([event.xdata, event.ydata, event.zdata])
        
        # Get current neuron coordinates
        mask = self.parcellation == self.current_neuron
        coords = np.array(np.where(mask)).T
        
        # Find nearest point to click
        distances = np.linalg.norm(coords - clicked, axis=1)
        nearest_idx = np.argmin(distances)
        
        # Toggle marker for nearest point
        point = tuple(coords[nearest_idx])
        self.markers[point] = ~self.markers[point]
        
        # Update plot
        self._update_plot()
    
    def _prev(self, event):
        """Go to previous neuron."""
        if self.current_neuron > 1:
            self.current_neuron -= 1
            self._update_plot()
    
    def _next(self, event):
        """Go to next neuron."""
        if self.current_neuron < self.max_neurons:
            self.current_neuron += 1
            self._update_plot()
    
    def _save(self, event):
        """Save current markers."""
        self.save_markers()
    
    def run(self):
        """Run the interactive pruning session."""
        if self.fig is None:
            self.setup_gui()
        plt.show()
    
    def save_markers(self):
        """Save the pruning markers."""
        with open("markers.pkl", "wb") as f:
            pickle.dump(self.markers, f)

