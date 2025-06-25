from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import tifffile
import os
from scipy import ndimage
from skimage.filters import gaussian, median
from skimage.feature import hessian_matrix
from sklearn.cluster import KMeans
import pickle
from pathlib import Path

@dataclass
class SegmentationConfig:
    """Configuration parameters for the segmentation process"""
    size_lim: List[int] = None  # Maximum crop size [x, y, z]
    clusters: int = 6           # Number of clusters for k-means
    sigma_gaussian: List[float] = None  # Gaussian smoothing parameters
    safe_margin: int = 3        # Safety margin for overlapping crops
    safe_margin_xy: int = 3     # Safety margin for xy plane
    background_threshold: float = None  # Threshold for background removal

class NeuronSegmentation:
    """Core class for 3D neuron segmentation from confocal microscopy images"""
    
    def __init__(self, config: SegmentationConfig = None):
        self.config = config or SegmentationConfig()
        self.image_info = {
            'dimensions': None,  # (Nx, Ny, Nz)
            'resolution': None,  # (x_res, y_res, z_res)
            'bit_depth': None,
            'dtype': None
        }
        self.crops_info = {}
        self._initialize_default_config()

    def _image_crops(self) -> List[Dict[str, int]]:


    def _initialize_default_config(self):
        """Initialize default configuration if not provided"""
        if not self.config.size_lim:
            self.config.size_lim = [1024, 1024, 10]
        if not self.config.sigma_gaussian:
            self.config.sigma_gaussian = [0]  # Default for 40x images

    def load_image(self, path: Union[str, Path], filename: str) -> None:
        """
        Load and analyze the input image file
        
        Args:
            path: Path to the image file
            filename: Name of the image file
        """
        full_path = os.path.join(path, filename)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f'Input file not found: {full_path}')

        with tifffile.TiffFile(full_path) as tif:
            # Get image dimensions and properties
            img_array = tif.asarray()
            self.image_info['dimensions'] = (
                img_array.shape[2],  # Nx
                img_array.shape[1],  # Ny
                img_array.shape[0]   # Nz
            )
            self.image_info['dtype'] = img_array.dtype
            self.image_info['bit_depth'] = 16 if img_array.dtype == np.uint16 else 8
            
            # Extract resolution information
            self._extract_resolution_info(tif.pages[0].tags)
            
            # Set background threshold based on bit depth
            if not self.config.background_threshold:
                self.config.background_threshold = 0.02 * (2 ** self.image_info['bit_depth'])

    def _extract_resolution_info(self, tags) -> None:
        """Extract resolution information from image tags"""
        res = [1, 1, 1]  # Default resolution
        
        # Extract XY resolution
        for i, tag in enumerate(['XResolution', 'YResolution']):
            if tag in tags:
                res[i] = 1 / (tags[tag].value[0] / tags[tag].value[1])
        
        # Extract Z resolution from ImageDescription if available
        if 'ImageDescription' in tags:
            desc = tags['ImageDescription'].value
            z_spacing = desc.find('spacing=')
            if z_spacing != -1:
                import re
                z_res = float(re.search(r'\d+\.\d+', desc[z_spacing + 8:]).group(0))
                res[2] = z_res
        
        self.image_info['resolution'] = tuple(res)

    def process_crop(self, image_crop: np.ndarray, crop_info: Dict[str, int]) -> Dict[str, np.ndarray]:
        """
        Process a single image crop
        
        Args:
            image_crop: 3D numpy array of the image crop
            crop_info: Dictionary containing crop coordinates and dimensions
        
        Returns:
            Dictionary containing processed crop data
        """
        # Apply median filter if specified
        if self.config.sigma_gaussian[0] == -1:
            image_crop = median(image_crop, size=3)

        # Calculate derivatives
        derivatives = self._calculate_derivatives(image_crop)
        
        # Perform k-means clustering
        clustering_result = self._perform_clustering(
            image_crop, 
            derivatives,
            self.config.background_threshold
        )

        return {
            'crop_data': image_crop,
            'derivatives': derivatives,
            'clustering': clustering_result
        }

    def _calculate_derivatives(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate first and second order derivatives"""
        # First order derivatives
        if self.config.sigma_gaussian[0] > 0:
            smoothed = gaussian(
                image.astype(np.float32),
                sigma=[
                    self.config.sigma_gaussian[0],
                    self.config.sigma_gaussian[0] * self.image_info['resolution'][0] / self.image_info['resolution'][1],
                    self.config.sigma_gaussian[0] * self.image_info['resolution'][0] / self.image_info['resolution'][2]
                ]
            )
            Gx, Gy, Gz = np.gradient(smoothed)
        else:
            Gx, Gy, Gz = np.gradient(image.astype(np.float32))

        # Second order derivatives (Hessian)
        Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = hessian_matrix(smoothed if self.config.sigma_gaussian[0] > 0 else image.astype(np.float32), sigma=1)

        return {
            'Gxx': Hxx,
            'Gyy': Hyy,
            'Gzz': Hzz,
            'Gx': Gx,
            'Gy': Gy,
            'Gz': Gz
        }

    def _perform_clustering(self, image: np.ndarray, derivatives: Dict[str, np.ndarray], 
                          threshold: float) -> np.ndarray:
        """Perform k-means clustering on the image features"""
        # Create feature space
        mask = image.ravel() >= threshold
        features = np.column_stack((
            image.ravel()[mask].astype(np.float32),
            derivatives['Gxx'].ravel()[mask],
            derivatives['Gyy'].ravel()[mask],
            derivatives['Gzz'].ravel()[mask]
        ))

        # Perform k-means clustering
        kmeans = KMeans(
            n_clusters=self.config.clusters,
            n_init=10,
            max_iter=1000,
            random_state=42
        )
        
        # Get cluster labels and reshape to original dimensions
        labels = np.ones(len(image.ravel()), dtype=np.uint8)
        labels[mask] = kmeans.fit_predict(features).astype(np.uint8) + 1
        
        return labels.reshape(image.shape)

    def refine_segmentation(self, segmentation: np.ndarray, 
                          original_image: np.ndarray) -> np.ndarray:
        """
        Refine the segmentation results
        
        Args:
            segmentation: Initial segmentation result
            original_image: Original image data
        
        Returns:
            Refined segmentation
        """
        # Fill 3D holes
        filled = self._fill_3d_holes(segmentation)
        
        # Apply intensity-based refinement
        intensity_threshold = 0.55 * (2 ** self.image_info['bit_depth'])
        refined = np.zeros_like(filled, dtype=bool)
        
        for z in range(filled.shape[2]):
            slice_seg = ndimage.binary_fill_holes(filled[:,:,z])
            slice_intensity = original_image[:,:,z] > intensity_threshold
            slice_combined = slice_seg | slice_intensity
            refined[:,:,z] = ndimage.binary_opening(slice_combined)
        
        return refined

    def _fill_3d_holes(self, binary_image: np.ndarray) -> np.ndarray:
        """Fill holes in 3D binary image"""
        filled = np.zeros_like(binary_image, dtype=bool)
        
        # Fill holes in each dimension
        for z in range(binary_image.shape[2]):
            filled[:,:,z] = ndimage.binary_fill_holes(binary_image[:,:,z])
        
        for y in range(binary_image.shape[1]):
            filled[:,y,:] = ndimage.binary_fill_holes(filled[:,y,:])
        
        for x in range(binary_image.shape[0]):
            filled[x,:,:] = ndimage.binary_fill_holes(filled[x,:,:])
        
        return filled

    def save_results(self, results: Dict[str, np.ndarray], 
                    output_path: Union[str, Path]) -> None:
        """Save segmentation results"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / 'segmentation_results.pkl', 'wb') as f:
            pickle.dump(results, f)
