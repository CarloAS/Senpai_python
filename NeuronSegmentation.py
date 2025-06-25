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
from skimage import measure
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

@dataclass
class SegmentationConfig:
    """Configuration parameters for the segmentation process"""
    size_lim: List[int] = None  # Maximum crop size [x, y, z]
    clusters: int = 6           # Number of clusters for k-means
    sigma_gaussian: List[float] = None  # Gaussian smoothing parameters
    safe_margin: int = 3        # Safety margin for overlapping crops
    safe_margin_xy: int = 3     # Safety margin for xy plane
    background_threshold: float = None  # Threshold for background removal

class NeuronSegmenter:
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
        self.initialize_default_config()
    
    def initialize_default_config(self):
        """Initialize default configuration if not provided"""
        if not self.config.size_lim:
            self.config.size_lim = self.image_info['dimensions']
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
            self.image_info['dimensions'] = [
                img_array.shape[2],  # Nx
                img_array.shape[1],  # Ny
                img_array.shape[0]   # Nz
            ]
            bit = img_array.dtype
            if bit == 'uint16': 
                self.image_info['bit_depth'] = 16
            elif bit == 'uint8':
                self.image_info['bit_depth'] = 8
            else:
                raise ValueError('input image must be 8 or 16 bit')
            
            # Extract resolution information
            self.extract_resolution_info(tif.pages[0].tags)
            
            # Set background threshold based on bit depth
            if not self.config.background_threshold:
                self.config.background_threshold = 0.02 * (2 ** self.image_info['bit_depth'])
            
            self.image_data = img_array.transpose(1, 2, 0)
            img = self.image_data
            return img

    def extract_resolution_info(self, tags) -> None:
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

    def voxelize_image(self) -> None:
        """Voxelize image data with size_lim"""
        Nx, Ny, Nz = self.image_info['dimensions']
        win = self.config.size_lim[2]  # Dimensione del ritaglio lungo l'asse z
        k_seq = np.arange(0, Nz, win)  # Numero di fette dell'immagine iniziale lungo l'asse z
        nxiL = np.arange(0, Nx, self.config.size_lim[0])
        nxeL = nxiL + self.config.size_lim[0]
        nyiL = np.arange(0, Ny, self.config.size_lim[1])
        nyeL = nyiL + self.config.size_lim[1]
        self.th_back = self.config.background_threshold
        NxC = nxeL - nxiL
        NyC = nyeL - nyiL
        nxiS = np.maximum(0, nxiL - self.config.safe_margin_xy)
        nxeS = np.minimum(Nx, nxeL + self.config.safe_margin_xy)
        nyiS = np.maximum(0, nyiL - self.config.safe_margin_xy)
        nyeS = np.minimum(Ny, nyeL + self.config.safe_margin_xy)
        zinit = np.maximum(0, k_seq - self.config.safe_margin)
        zend = np.minimum(Nz, k_seq + win - self.config.safe_margin)
        NzC = np.minimum(Nz, k_seq + win) - k_seq
        voxels = {}

        for i,x in enumerate(nxiL):
            for j,y in enumerate(nyiL):
                for k in k_seq:
                    voxels[f'crop_{x}_{y}_{k}'] = {
                        'nxi': nxiL[i],
                        'nxe': nxeL[i],
                        'nyi': nyiL[j],
                        'nye': nyeL[j],
                        'k': k,
                        'nxiS': nxiS[i],
                        'nxeS': nxeS[i],
                        'nyiS': nyiS[j],
                        'nyeS': nyeS[j],
                        'zinit': zinit[k],
                        'zend': zend[k],
                        'NxC': NxC[i],
                        'NyC': NyC[j],
                        'NzC': NzC[k],
                        'cIMc': self.image_data[nxiS[i]:nxeS[i], nyiS[j]:nyeS[j], zinit[k]:zend[k]],
                        'reshaped_cIMc': None,
                        'GxxKt': None,
                        'GyyKt': None,
                        'GzzKt': None,
                        'GxxKt_2': None,
                        'GyyKt_2': None,
                        'GzzKt_2': None,
                        'clustering': None,
                        'clustering_2': None,
                        'TOT_KM1': None,
                        'TOT_KM2': None
                    }
                    
        self.k_seq = k_seq
        self.win = win
    
        return voxels

    def k_means_voxels(self):
        """Perform k-means clustering on each voxel"""
        voxels = self.voxelize_image()
        sig_G = self.config.sigma_gaussian

        for voxel in voxels.values():

            derivatives = self.process_derivatives(voxel)

            voxel = self.reshape_voxel(voxel, derivatives)

            TOT_KM1, TOT_KM2 = self.kmeans_clustering(voxel,derivatives)

            voxel['TOT_KM1'] = TOT_KM1

        return voxels
    
    
    def process_derivatives(self,voxel):
        """Analyze the cropped image"""
        derivatives = {
            'Gxx2': None,
            'Gyy2': None,
            'Gzz2': None,
            'Gxx_2': None,
            'Gyy_2': None,
            'Gzz_2': None
        }
        cIMc = voxel['cIMc']
        # 143,256,143
        sig_G = self.config.sigma_gaussian
        resolution = self.image_info['resolution']
        # Filtro mediano se sig_G[0] == -1
        if sig_G[0] == -1:
            print('applying 3x3x3 median filter...')
            cIMc = median(cIMc, size=3)

        # Fornisce feedback all'utente
        print(f'1st of {len(sig_G)} passes...')

        # Derivate di primo ordine
        if sig_G[0] > 0:
            cIMc_smoothed = gaussian(cIMc.astype(np.float32), 
                                     sigma=[sig_G[0], sig_G[0] * resolution[0] / resolution[1], sig_G[0] * resolution[0] / resolution[2]])
            Gx, Gy, Gz = np.gradient(cIMc_smoothed)
        else:
            Gx, Gy, Gz = np.gradient(cIMc.astype(np.float32))

        # Derivate di secondo ordine
        Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = hessian_matrix(cIMc_smoothed if sig_G[0] > 0 else cIMc.astype(np.float32), sigma=1)
        

        if len(sig_G)>2:
            cIMc_smoothed = gaussian(cIMc.astype(np.float32), 
                                     sigma=[sig_G[3], sig_G[3] * resolution[0] / resolution[1], sig_G[0] * resolution[0] / resolution[2]])
            Gx, Gy, Gz = np.gradient(cIMc_smoothed)

        # Derivate di secondo ordine
            Hxx_2, Hxy_2, Hxz_2, Hyy_2, Hyz_2, Hzz_2 = hessian_matrix(cIMc_smoothed, sigma=1)

        derivatives = {
            'Gxx': Hxx,
            'Gyy': Hyy,
            'Gzz': Hzz,
            'Gxx_2': None if len(sig_G) <= 2 else Hxx_2,
            'Gyy_2': None if len(sig_G) <= 2 else Hyy_2,
            'Gzz_2': None if len(sig_G) <= 2 else Hzz_2
        }

        return derivatives
    
    def reshape_voxel(self, voxel, derivatives):
        """Reshape the image for k-means clustering"""
        cIMc = voxel['cIMc']
        safe_xy = self.config.safe_margin_xy
        safe_z = self.config.safe_margin
        Gxx = derivatives['Gxx']
        Gyy = derivatives['Gyy']
        Gzz = derivatives['Gzz']
        Gxx_2 = derivatives['Gxx_2']
        Gyy_2 = derivatives['Gyy_2']
        Gzz_2 = derivatives['Gzz_2']
        win = self.config.size_lim[2]
        nxi = voxel['nxi']
        nyi = voxel['nyi']
        NxC = voxel['NxC']
        NyC = voxel['NyC']
        k = voxel['k']

        voxel['reshaped_cIMc'] = cIMc[
            min(safe_xy,nxi):min(NxC+min(safe_xy,nxi), cIMc.shape[0]),
            min(safe_xy,nyi):min(NyC+min(safe_xy,nyi), cIMc.shape[1]),
            min(safe_z,k):min(win+min(safe_z,k), cIMc.shape[2]),
        ]
        voxel['GxxKt'] = Gxx[
            min(safe_xy,nxi):min(NxC+min(safe_xy,nxi), Gxx.shape[0]),
            min(safe_xy,nyi):min(NyC+min(safe_xy,nyi), Gxx.shape[1]),
            min(safe_z,k):min(win+min(safe_z,k), Gxx.shape[2]),
        ]

        voxel['GyyKt'] = Gyy[
            min(safe_xy,nxi):min(NxC+min(safe_xy,nxi), Gyy.shape[0]),
            min(safe_xy,nyi):min(NyC+min(safe_xy,nyi), Gyy.shape[1]),
            min(safe_z,k):min(win+min(safe_z,k), Gyy.shape[2]),
        ]

        voxel['GzzKt'] = Gzz[
            min(safe_xy,nxi):min(NxC+min(safe_xy,nxi), Gzz.shape[0]),
            min(safe_xy,nyi):min(NyC+min(safe_xy,nyi), Gzz.shape[1]),
            min(safe_z,k):min(win+min(safe_z,k), Gzz.shape[2]),
        ]

        if Gxx_2 is not None:
                
            voxel['GxxKt_2'] = Gxx_2[
                min(safe_xy,nxi):min(NxC+min(safe_xy,nxi), Gxx_2.shape[0]),
                min(safe_xy,nyi):min(NyC+min(safe_xy,nyi), Gxx_2.shape[1]),
                min(safe_z,k):min(win+min(safe_z,k), Gxx_2.shape[2]),
            ]

            voxel['GyyKt_2'] = Gyy_2[
                min(safe_xy,nxi):min(NxC+min(safe_xy,nxi), Gyy_2.shape[0]),
                min(safe_xy,nyi):min(NyC+min(safe_xy,nyi), Gyy_2.shape[1]),
                min(safe_z,k):min(win+min(safe_z,k), Gyy_2.shape[2]),
            ]

            voxel['GzzKt_2'] = Gzz_2[
                min(safe_xy,nxi):min(NxC+min(safe_xy,nxi), Gzz_2.shape[0]),
                min(safe_xy,nyi):min(NyC+min(safe_xy,nyi), Gzz_2.shape[1]),
                min(safe_z,k):min(win+min(safe_z,k), Gzz_2.shape[2]),
            ]    

        return voxel

    def kmeans_clustering(self, voxel, derivatives):
        """Perform k-means clustering on the image features"""
        cIMc = voxel['reshaped_cIMc']
        GxxKt = voxel['GxxKt']
        GyyKt = voxel['GyyKt']
        GzzKt = voxel['GzzKt']
        NxC = voxel['NxC']
        NyC = voxel['NyC']
        NzC = voxel['NzC']
        tp = self.image_info['dtype']
        GxxKt_2 = voxel['GxxKt_2']
        GyyKt_2 = voxel['GyyKt_2']
        GzzKt_2 = voxel['GzzKt_2']
        mask = cIMc.ravel() >= self.th_back
        resiz_km = np.ones(len(cIMc.ravel()), dtype=np.uint8)

        print(cIMc.shape)
        # Definisci lo spazio delle caratteristiche per il k-means
        km_in1 = np.column_stack((
            cIMc.ravel()[mask].astype(np.float32),
            GxxKt.ravel()[mask],
            GyyKt.ravel()[mask],
            GzzKt.ravel()[mask]
        ))

        # Initialize TOT_KM1 with proper shape
        TOT_KM1 = np.ones((NxC, NyC, NzC), dtype=tp)

        # Clustering k-means
        kmeans = KMeans(n_clusters=self.config.clusters, n_init=10, max_iter=1000, random_state=42)
        labels = kmeans.fit_predict(km_in1.astype(np.float32))
        resiz_km[mask] = labels.astype(np.uint8)
        TOT_KM1 = resiz_km.reshape((NxC, NyC, NzC))

        TOT_KM2 = None

        if GxxKt_2 is not None:
            resiz_km = np.ones(len(cIMc.ravel()), dtype=np.uint8)

            # Definisci lo spazio delle caratteristiche per il k-means
            km_in1 = np.column_stack((
                cIMc.ravel()[mask].astype(np.float32),
                GxxKt_2.ravel()[mask],
                GyyKt_2.ravel()[mask],
                GzzKt_2.ravel()[mask]
            ))

            # Initialize TOT_KM1 with proper shape
            TOT_KM2 = np.ones((NxC, NyC, NzC), dtype=tp)

            # Clustering k-means
            kmeans = KMeans(n_clusters=self.config.clusters, n_init=10, max_iter=1000, random_state=42)
            labels = kmeans.fit_predict(km_in1.astype(np.float32))
            resiz_km[mask] = labels.astype(np.uint8)
            TOT_KM2 = resiz_km.reshape((NxC, NyC, NzC))

        return TOT_KM1, TOT_KM2

    def process_segmentation(self, voxels):
        
        senpai_KM_lv1, senpai_KM_lv2, cIM = self.recompose(voxels)

        mask_lv1, mask_lv2 = self.compute_neuron_mask(senpai_KM_lv1, senpai_KM_lv2, cIM, voxels)

        refined_seg_1, refined_seg_2 = self.advanced_segmentation_refinement(senpai_KM_lv1,senpai_KM_lv2,cIM, mask_lv1, mask_lv2)

        return senpai_KM_lv1, senpai_KM_lv2, refined_seg_1, refined_seg_2


    def recompose(self, voxels):
        """Recompose the segmented image from individual crops"""
        Nx, Ny, Nz = self.image_info['dimensions']
        win = self.win
        tp = self.image_info['dtype']
        senpai_KM_lv1 = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
        senpai_KM_lv2 = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
        cIM = np.zeros((Nx, Ny, Nz), dtype=tp)
        sig_G = self.config.sigma_gaussian
        
        for voxel in voxels.values():
            nxi = voxel['nxi']
            nxe = voxel['nxe']
            nyi = voxel['nyi']
            nye = voxel['nye']
            k = voxel['k']
            senpai_KM_lv1[nxi:nxe, nyi:nye, k:k+win] = voxel['TOT_KM1']
            cIM[nxi:nxe, nyi:nye, k:k+win] = voxel['reshaped_cIMc']
            if len(sig_G) > 2:
                senpai_KM_lv2[nxi:nxe, nyi:nye, k:k+win] = voxel['TOT_KM2']

        if len(sig_G) <= 2:
            senpai_KM_lv2 = None

        return senpai_KM_lv1, senpai_KM_lv2, cIM

    def compute_neuron_mask(self,senpai_KM_lv1, senpai_KM_lv2, cIM, voxels):
        """
        Compute neuron masks by analyzing second-order derivatives within each k-means cluster.
        Creates binary masks (senpai_KM_lv1_msk and senpai_KM_lv2_msk) identifying neural structures.
        """
        Nx, Ny, Nz = self.image_info['dimensions']
        win = self.win
        clusters = self.config.clusters
        size_lim = self.config.size_lim
        sig_G = self.config.sigma_gaussian
        
        # Initialize masks
        mask_lv1 = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
        
        # Process each voxel for first level mask
        for k in range(0, Nz, win):
            for x in range(0, Nx, size_lim[0]):
                for y in range(0, Ny, size_lim[1]):
                    # Calculate bounds
                    x_end = min(x + size_lim[0], Nx)
                    y_end = min(y + size_lim[1], Ny)
                    k_end = min(k + win, Nz)
                    
                    # Initialize arrays for cluster means
                    Gxx_cl = np.zeros(clusters)
                    Gyy_cl = np.zeros(clusters)
                    Gzz_cl = np.zeros(clusters)
                    
                    # Get current voxel data
                    voxel_key = f'crop_{x}_{y}_{k}'
                    current_voxel = voxels[voxel_key]
                    
                    # Calculate mean values for each cluster
                    for clu in range(clusters):
                        cluster_mask = senpai_KM_lv1[x:x_end, y:y_end, k:k_end] == clu
                        if np.any(cluster_mask):
                            Gxx_cl[clu] = np.mean(current_voxel['GxxKt'][cluster_mask])
                            Gyy_cl[clu] = np.mean(current_voxel['GyyKt'][cluster_mask])
                            Gzz_cl[clu] = np.mean(current_voxel['GzzKt'][cluster_mask])
                    
                    # Find last index where condition is not met (add 1 for 1-based cluster numbering)
                    last_index = np.where((Gxx_cl < 0) & (Gyy_cl < 0) & (Gzz_cl < 0) == 0)[0][-1] +1
                    
                    # Assign mask values
                    mask_lv1[x:x_end, y:y_end, k:k_end] = last_index
                    
        mask_lv2 = None
        # Process second level mask if multiple smoothing parameters are specified
        if len(sig_G) > 1:
            mask_lv2 = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
            
            for k in range(0, Nz, win):
                for x in range(0, Nx, size_lim[0]):
                    for y in range(0, Ny, size_lim[1]):
                        # Calculate bounds
                        x_end = min(x + size_lim[0], Nx)
                        y_end = min(y + size_lim[1], Ny)
                        k_end = min(k + win, Nz)
                        
                        # Initialize arrays for cluster means
                        Gxx_cl = np.zeros(clusters)
                        Gyy_cl = np.zeros(clusters)
                        Gzz_cl = np.zeros(clusters)
                        
                        # Get current voxel data
                        voxel_key = f'crop_{x}_{y}_{k}'
                        current_voxel = voxels[voxel_key]
                        
                        # Calculate mean values for each cluster
                        for clu in range(clusters):
                            cluster_mask = senpai_KM_lv2[x:x_end, y:y_end, k:k_end] == clu
                            if np.any(cluster_mask):
                                Gxx_cl[clu] = np.mean(current_voxel['GxxKt_2'][cluster_mask])
                                Gyy_cl[clu] = np.mean(current_voxel['GyyKt_2'][cluster_mask])
                                Gzz_cl[clu] = np.mean(current_voxel['GzzKt_2'][cluster_mask])
                        
                        # Find last index where condition is not met (add 1 for 1-based cluster numbering)
                        last_index = np.where((Gxx_cl < 0) & (Gyy_cl < 0) & (Gzz_cl < 0) == 0)[0][-1] + 1
                        
                        # Assign mask values
                        mask_lv2[x:x_end, y:y_end, k:k_end] = last_index

        return mask_lv1, mask_lv2

    def advanced_segmentation_refinement(self, initial_seg_1, initial_seg_2, original_image, mask_lv1, mask_lv2):
        """
        Refine segmentation using computed mask levels and various morphological operations
        
        Parameters:
        -----------
        initial_seg_1 : ndarray
            Initial k-means segmentation (level 1)
        initial_seg_2 : ndarray
            Initial k-means segmentation (level 2) or None
        original_image : ndarray
            Original intensity image
        mask_lv1 : ndarray
            First level mask computed from derivative analysis
        mask_lv2 : ndarray
            Second level mask computed from derivative analysis or None
            
        Returns:
        --------
        refined_seg_1 : ndarray
            Refined segmentation from level 1 analysis
        refined_seg_2 : ndarray
            Refined segmentation from level 2 analysis or None
        """
        bit_depth = self.image_info['bit_depth']
        # Compute threshold based on bit depth
        intensity_threshold = 0.55 * (2 ** bit_depth)
        
        # Create binary masks based on the cluster indices in mask_lv1
        # The mask_lv1 contains the last index of clusters that don't meet neuron criteria
        # We want to keep all clusters above this threshold
        binary_mask_1 = np.zeros_like(initial_seg_1, dtype=bool)
        for z in range(initial_seg_1.shape[2]):
            for y in range(initial_seg_1.shape[1]):
                for x in range(initial_seg_1.shape[0]):
                    if initial_seg_1[x, y, z] >= mask_lv1[x, y, z]:
                        binary_mask_1[x, y, z] = True
        
        # 3D hole filling on the binary mask
        filled_seg_1 = self._imfill_3d(binary_mask_1, original_image, bit_depth)
        
        # Additional segmentation refinement
        refined_seg_1 = np.zeros_like(filled_seg_1, dtype=bool)
        
        # Process each slice
        for z in range(filled_seg_1.shape[2]):
            # Fill holes in 2D slice
            slice_seg = ndimage.binary_fill_holes(filled_seg_1[:,:,z])
            
            # Additional intensity-based segmentation
            slice_intensity = original_image[:,:,z] > intensity_threshold
            slice_combined = slice_seg | slice_intensity
            
            # Remove small artifacts
            slice_cleaned = ndimage.binary_opening(slice_combined)
            refined_seg_1[:,:,z] = slice_cleaned
        
        # Apply final 3D morphological operations
        refined_seg_1 = ndimage.binary_closing(refined_seg_1, structure=np.ones((3,3,3)))
        
        # Initialize refined_seg_2
        refined_seg_2 = None
        
        # Process second level mask if available
        if initial_seg_2 is not None and mask_lv2 is not None:
            binary_mask_2 = np.zeros_like(initial_seg_2, dtype=bool)
            for z in range(initial_seg_2.shape[2]):
                for y in range(initial_seg_2.shape[1]):
                    for x in range(initial_seg_2.shape[0]):
                        if initial_seg_2[x, y, z] >= mask_lv2[x, y, z]:
                            binary_mask_2[x, y, z] = True
            
            # 3D hole filling
            filled_seg_2 = self._imfill_3d(binary_mask_2, original_image, bit_depth)
            
            # Additional segmentation refinement
            refined_seg_2 = np.zeros_like(filled_seg_2, dtype=bool)
            
            for z in range(filled_seg_2.shape[2]):
                # Fill holes in 2D slice
                slice_seg = ndimage.binary_fill_holes(filled_seg_2[:,:,z])
                
                # Additional intensity-based segmentation
                slice_intensity = original_image[:,:,z] > intensity_threshold
                slice_combined = slice_seg | slice_intensity
                
                # Remove small artifacts
                slice_cleaned = ndimage.binary_opening(slice_combined)
                refined_seg_2[:,:,z] = slice_cleaned
            
            # Apply final 3D morphological operations
            refined_seg_2 = ndimage.binary_closing(refined_seg_2, structure=np.ones((3,3,3)))
        
        return refined_seg_1, refined_seg_2

    def _imfill_3d(self, binary_image, original_image, bit_depth=8):
        """3D hole filling using scipy.ndimage with intensity thresholding in all dimensions
        
        Parameters:
        -----------
        binary_image : ndarray
            3D binary segmentation image where holes need to be filled
        original_image : ndarray
            Original grayscale image used for thresholding
        bit_depth : int
            Bit depth of the original image (default: 8)
            
        Returns:
        --------
        filled : ndarray
            3D binary image with holes filled
        """
        # Initialize output array with initial segmentation
        filled = np.array(binary_image, dtype=bool)
        
        # Calculate intensity threshold
        threshold = 0.55 * (2 ** bit_depth)
        
        # Fill holes in each 2D slice along z-axis
        for z in range(binary_image.shape[2]):
            high_intensity_mask = (filled[:,:,z] == 0) & (original_image[:,:,z] > threshold)
            combined_mask = filled[:,:,z] | high_intensity_mask
            filled[:,:,z] = ndimage.binary_fill_holes(combined_mask)
        
        # Fill holes in each 2D slice along y-axis
        for y in range(binary_image.shape[1]):
            high_intensity_mask = (filled[:,y,:] == 0) & (original_image[:,y,:] > threshold)
            combined_mask = filled[:,y,:] | high_intensity_mask
            filled[:,y,:] = ndimage.binary_fill_holes(combined_mask)
        
        # Fill holes in each 2D slice along x-axis
        for x in range(binary_image.shape[0]):
            high_intensity_mask = (filled[x,:,:] == 0) & (original_image[x,:,:] > threshold)
            combined_mask = filled[x,:,:] | high_intensity_mask
            filled[x,:,:] = ndimage.binary_fill_holes(combined_mask)
        
        return filled
        

    def visualize_kmeans_with_slider(self,k_means):
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)  # Make room for the slider
        
        # Get initial slice (middle of the volume)
        z_slice = k_means.shape[2] // 2
        
        # Display the initial slice
        img = ax.imshow(k_means[:, :, z_slice], cmap='viridis')
        plt.colorbar(img, ax=ax, label='Cluster')
        ax.set_title(f'K-means clustering (z-slice: {z_slice})')
        
        # Add slider axis
        ax_slider = plt.axes([0.2, 0.05, 0.65, 0.04])
        
        # Create the slider
        slider = Slider(
            ax=ax_slider,
            label='Z-Slice',
            valmin=0,
            valmax=k_means.shape[2] - 1,
            valinit=z_slice,
            valstep=1,
        )
        
        # Update function for the slider
        def update(val):
            new_slice = int(slider.val)
            img.set_data(k_means[:, :, new_slice])
            ax.set_title(f'K-means clustering (z-slice: {new_slice})')
            fig.canvas.draw_idle()
        
        # Register the update function with the slider
        slider.on_changed(update)
        
        plt.show()
        
        return fig, ax, slider
    
    def visualize_3d_segmentation(self, segmentation, threshold=0.5, fig_size=(10, 8), 
                             alpha=0.5, color='red', view_angle=(30, 30)):
        """
        Create a 3D visualization of a binary segmentation using matplotlib.
        
        Parameters:
        -----------
        segmentation : ndarray
            3D binary segmentation array
        threshold : float
            Isosurface threshold value (default: 0.5)
        fig_size : tuple
            Figure size (width, height) in inches
        alpha : float
            Transparency of the surface (0-1)
        color : str
            Color of the segmentation surface
        view_angle : tuple
            Initial viewing angle (elevation, azimuth) in degrees
            
        Returns:
        --------
        fig : matplotlib figure
            The figure containing the 3D visualization
        """
        # Convert to binary if not already
        """if not segmentation.dtype == bool:
            segmentation = segmentation > threshold"""
        
        # Use marching cubes algorithm to get the surface mesh
        verts, faces, normals, values = measure.marching_cubes(segmentation, level=0.5)
        
        # Create a new figure with 3D axes
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the surface
        mesh = ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                            triangles=faces, color=color, alpha=alpha, 
                            edgecolor='none')
        
        # Set equal aspect ratio
        # Get the limits of the plot
        x_lim = ax.get_xlim3d()
        y_lim = ax.get_ylim3d()
        z_lim = ax.get_zlim3d()
        
        # Calculate the range of each dimension
        x_range = abs(x_lim[1] - x_lim[0])
        y_range = abs(y_lim[1] - y_lim[0])
        z_range = abs(z_lim[1] - z_lim[0])
        
        # Find the maximum range
        max_range = max(x_range, y_range, z_range) / 2
        
        # Set the limits of each axis based on the maximum range
        mid_x = (x_lim[1] + x_lim[0]) / 2
        mid_y = (y_lim[1] + y_lim[0]) / 2
        mid_z = (z_lim[1] + z_lim[0]) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Set the view angle
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Visualization of Segmentation')
        
        return fig

    def compare_segmentations(self, original_seg, refined_seg, alpha=0.5, fig_size=(15, 8)):
        """
        Create a side-by-side comparison of original and refined segmentations.
        
        Parameters:
        -----------
        original_seg : ndarray
            3D binary array of original segmentation
        refined_seg : ndarray
            3D binary array of refined segmentation
        alpha : float
            Transparency of the surfaces (0-1)
        fig_size : tuple
            Figure size (width, height) in inches
            
        Returns:
        --------
        fig : matplotlib figure
            The figure containing the comparison visualization
        """
        # Create a new figure with two 3D axes
        fig = plt.figure(figsize=fig_size)
        
        # Original segmentation
        ax1 = fig.add_subplot(121, projection='3d')
        verts1, faces1, _, _ = measure.marching_cubes(original_seg, level=0.5)
        mesh1 = ax1.plot_trisurf(verts1[:, 0], verts1[:, 1], verts1[:, 2],
                            triangles=faces1, color='blue', alpha=alpha, 
                            edgecolor='none')
        ax1.set_title('Original Segmntation')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Refined segmentation
        ax2 = fig.add_subplot(122, projection='3d')
        verts2, faces2, _, _ = measure.marching_cubes(refined_seg, level=0.5)
        mesh2 = ax2.plot_trisurf(verts2[:, 0], verts2[:, 1], verts2[:, 2],
                            triangles=faces2, color='red', alpha=alpha, 
                            edgecolor='none')
        ax2.set_title('Refined Segmentation')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Set the same view angle for both plots
        ax1.view_init(elev=30, azim=30)
        ax2.view_init(elev=30, azim=30)
        
        # Adjust layout
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def visualize_segmentation_comparison_xyz(self, original_image, binary_segmentation, fig_title="Segmentation Comparison (X, Y, Z planes)"):
        """
        Visualizes a 3D binary segmentation slice by slice alongside the original image,
        assuming both inputs have dimensions (X, Y, Z).

        Args:
            original_image (np.ndarray): The original 3D image (Nx, Ny, Nz).
            binary_segmentation (np.ndarray): The 3D binary segmentation result (Nx, Ny, Nz).
            fig_title (str): The title for the matplotlib figure.
        """

        #if len(np.shape(original_image)) != 3 or binary_segmentation.ndim != 3:
        #    raise ValueError("Both images must be 3D arrays with shape (Nx, Ny, Nz).")

        # Expected shape: (Nx, Ny, Nz)
        nx_orig, ny_orig, nz_orig = original_image.shape
        nx_seg, ny_seg, nz_seg = binary_segmentation.shape

        if nx_orig != nx_seg or ny_orig != ny_seg:
            raise ValueError(
                f"X and Y dimensions of original image ({nx_orig}, {ny_orig}) "
                f"and segmentation ({nx_seg}, {ny_seg}) must match."
            )

        # Ensure segmentation is boolean or can be interpreted as such
        if binary_segmentation.dtype != bool:
            # Assuming positive values indicate segmentation if not boolean
            binary_segmentation_display = binary_segmentation > 0
        else:
            binary_segmentation_display = binary_segmentation

        # Number of slices along Z-axis
        # If Z dimensions differ, we'll use the minimum to avoid errors, and warn the user.
        num_slices = min(nz_orig, nz_seg)
        if nz_orig != nz_seg:
            print(f"Warning: Z dimension of original image ({nz_orig}) and segmentation ({nz_seg}) "
                f"do not match. Displaying up to slice {num_slices -1}.")


        if num_slices == 0:
            raise ValueError("Number of Z-slices is zero. Cannot visualize.")

        # Create the figure and subplots
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        plt.subplots_adjust(bottom=0.25)  # Make room for the slider and title

        fig.suptitle(fig_title, fontsize=16)

        # Get initial slice (middle of the Z volume)
        initial_z_slice = num_slices // 2

        # Display the initial XY-planes (slices along Z)
        # original_image[X, Y, Z_slice]
        img_orig = axs[0].imshow(original_image[:, :, initial_z_slice], cmap='gray', origin='lower')
        axs[0].set_title(f'Original Image (Z-Slice: {initial_z_slice})')
        axs[0].set_xlabel('Y-axis')
        axs[0].set_ylabel('X-axis')
        axs[0].axis('on') # Keep axes to show X and Y context

        img_seg = axs[1].imshow(binary_segmentation_display[:, :, initial_z_slice], cmap='viridis', origin='lower')
        axs[1].set_title(f'Segmentation (Z-Slice: {initial_z_slice})')
        axs[1].set_xlabel('Y-axis')
        axs[1].set_ylabel('X-axis')
        axs[1].axis('on')

        # Add slider axis
        ax_slider = plt.axes([0.2, 0.1, 0.65, 0.04], facecolor='lightgoldenrodyellow')

        # Create the slider for Z-slices
        slider = Slider(
            ax=ax_slider,
            label='Z-Slice Index',
            valmin=0,
            valmax=num_slices - 1,
            valinit=initial_z_slice,
            valstep=1,
        )

        # Update function for the slider
        def update(val):
            new_slice = int(slider.val)

            # Update original image plot
            if new_slice < nz_orig:
                img_orig.set_data(original_image[:, :, new_slice])
                axs[0].set_title(f'Original Image (Z-Slice: {new_slice})')
            else:
                # This case should ideally not be reached if num_slices is min(nz_orig, nz_seg)
                # but as a safeguard:
                axs[0].set_title(f'Original Image (Z-Slice: {new_slice} - Out of bounds)')
                img_orig.set_data(np.zeros((nx_orig, ny_orig))) # Show blank

            # Update segmentation plot
            if new_slice < nz_seg:
                img_seg.set_data(binary_segmentation_display[:, :, new_slice])
                axs[1].set_title(f'Segmentation (Z-Slice: {new_slice})')
            else:
                axs[1].set_title(f'Segmentation (Z-Slice: {new_slice} - Out of bounds)')
                img_seg.set_data(np.zeros((nx_seg, ny_seg))) # Show blank
                
            fig.canvas.draw_idle()

        # Register the update function with the slider
        slider.on_changed(update)

        plt.show()

        return fig, axs, slider

"""def advanced_segmentation_refinement(self, initial_seg_1, initial_seg_2, original_image, mask_lv1, masK_lv2):
"""
"""Refine segmentation by:
- Filling holes
- Applying intensity-based constraints
- Cleaning small clusters"""
"""
def imfill_3d(binary_image, original_image, bit_depth=8):
"""
"""3D hole filling using scipy.ndimage with intensity thresholding in all dimensions

Parameters:
-----------
binary_image : ndarray
3D binary segmentation image where holes need to be filled
original_image : ndarray
Original grayscale image used for thresholding
bit_depth : int
Bit depth of the original image (default: 8)

Returns:
--------
filled : ndarray
3D binary image with holes filled"""
"""
# Initialize output array with initial segmentation
filled = np.array(binary_image, dtype=bool)

# Calculate intensity threshold
threshold = 0.55 * (2 ** bit_depth)

# Fill holes in each 2D slice along z-axis
for z in range(binary_image.shape[2]):
high_intensity_mask = (filled[:,:,z] == 0) & (original_image[:,:,z] > threshold)
combined_mask = filled[:,:,z] | high_intensity_mask
filled[:,:,z] = ndimage.binary_fill_holes(combined_mask)

# Fill holes in each 2D slice along y-axis
for y in range(binary_image.shape[1]):
high_intensity_mask = (filled[:,y,:] == 0) & (original_image[:,y,:] > threshold)
combined_mask = filled[:,y,:] | high_intensity_mask
filled[:,y,:] = ndimage.binary_fill_holes(combined_mask)

# Fill holes in each 2D slice along x-axis
for x in range(binary_image.shape[0]):
high_intensity_mask = (filled[x,:,:] == 0) & (original_image[x,:,:] > threshold)
combined_mask = filled[x,:,:] | high_intensity_mask
filled[x,:,:] = ndimage.binary_fill_holes(combined_mask)

return filled

bit_depth = self.image_info['bit_depth']
# Compute threshold based on bit depth
intensity_threshold = 0.55 * (2 ** bit_depth)
refined_seg_2 = None

# 3D hole filling
filled_seg_1 = imfill_3d(initial_seg_1, self.image_data, bit_depth)
# Additional segmentation refinement
refined_seg_1 = np.zeros_like(filled_seg_1, dtype=bool)

# Process each slice and dimension
for z in range(filled_seg_1.shape[2]):
# Fill holes in 2D slice
slice_seg = ndimage.binary_fill_holes(filled_seg_1[:,:,z])

# Additional intensity-based segmentation
slice_intensity = original_image[:,:,z] > intensity_threshold
slice_combined = slice_seg | slice_intensity

# Remove small artifacts
slice_cleaned = ndimage.binary_opening(slice_combined)
refined_seg_1[:,:,z] = slice_cleaned

if initial_seg_2 is not None:
# 3D hole filling
filled_seg_2 = imfill_3d(initial_seg_1)
# Additional segmentation refinement
refined_seg_2 = np.zeros_like(filled_seg_2, dtype=bool)

for z in range(filled_seg_2.shape[2]):
# Fill holes in 2D slice
slice_seg = ndimage.binary_fill_holes(filled_seg_2[:,:,z])

# Additional intensity-based segmentation
slice_intensity = original_image[:,:,z] > intensity_threshold
slice_combined = slice_seg | slice_intensity

# Remove small artifacts
slice_cleaned = ndimage.binary_opening(slice_combined)
refined_seg_2[:,:,z] = slice_cleaned

return refined_seg_1, refined_seg_2"""