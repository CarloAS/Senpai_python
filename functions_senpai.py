import os
import tifffile
import numpy as np
import scipy.io as sio
from scipy import ndimage
import warnings
import scipy.ndimage as ndi
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import sys
import re
from skimage.filters import gaussian, median
from skimage import io
from skimage.segmentation import watershed
from skimage.feature import hessian_matrix
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

def senpai_seg_core_v4(path_in=None, im_in=None, *args):

    """
Mandatory inputs:
  path_in:  the path at which the file im_in is found.

  im_in:    is a string specifying the file name of the input
            image (.tiff file type, either 8 or 16 bits)

Elective inputs:
  path out (char):    path in which partial and final results are saved.
                      Default is ./senpaiseg/

  sig_G (numerical):  scalar or 1x2 numerical array.
                      The segmentation is repeated length(sig_G) times.
                      Each time, derivatives are computed on im_in smoothed
                      with a 3d Gaussian kernel having sigma = sig_G(p).
                      Default is sig_G=[0], advised for 40x images;
                      sig_G=[0 3] advised for 93x images.
                      If present, sig_G(2) must be > 0
                      sig_G(1)=-1 applies a 3x3x3 median filter

  size_lim (numerical 1x3 array):   maximum crop size on which the
                                    algorithm is allowed to work.
                                    The image will be divided in crops to save memory.
                                    size_lim should be set according to the specifics of your machine.
                                    Default is size_lim=[1024 1024 10];

   verbmem (boolean):    if 1, keep partial results (slabs-specific), else,
                         delete them. Default is 0.

    paralpool (boolean):  if 1, starts a parallel pool while
                         doing the kmeans. Default is 1.

    clusters (numerical): number of classes for kmeans clustering.
                          Default is 6
    """


    # Controlla gli argomenti di input
    if path_in is None or im_in is None:
        try:
            path_in = input("Please enter the input file path: ")
            im_in = input("Please enter the input file name: ")
            path_out = input("Please select the output directory: ")
        except:
            print('No dataset selected')
            return

    if path_in is None or im_in is None:
        raise ValueError('Not enough input arguments!')
    else:
        print(f'Input file: {os.path.join(path_in, im_in)}')
        if not os.path.isfile(os.path.join(path_in, im_in)):
            raise FileNotFoundError('Input file not found')

        # Raccogli informazioni dall'intestazione dell'immagine
        with tifffile.TiffFile('test_data.tif') as tif:
            # Get the image array
            info1 = tif.asarray()
        Nz = len(info1)
        Ny = info1[0].shape[1]
        Nx = info1[0].shape[0]
        tp = info1[0].dtype


        # Get the tags
        tags = tif.pages[0].tags

        res_tags = ['XResolution', 'YResolution']
        res = [1, 1, 1] # [xres, yres ,zres]
        for i,res_tag in enumerate(res_tags):
            if res_tag in tags:
                res[i] = 1 / (tags[res_tag].value[0] / tags[res_tag].value[1])
            else:
                res[i] = 1
        if 'ImageDescription' in tags:
            strinfo = tags['ImageDescription'].value
            um = strinfo.find('spacing=')
            cut_s = len('spacing=')
            res[2] = float(re.search(r'\d+\.\d+', strinfo[um + cut_s:]).group(0))
        else:
            res[2] = 1

        # Definizione di tp in base alla profondità di bit dell'immagine
        if tp == 'uint16': 
            bit = 16
        elif tp == 'uint8':
            bit = 8
        else:
            raise ValueError('input image must be 8 or 16 bit')

        # Impostazione dei valori predefiniti
        if len(args) == 0:
            try:
                path_out = input("Please select the output directory: ")
            except:
                path_out = os.path.join(os.getcwd(), 'senpaiseg')

        sig_G = [0]
        if Nx * Ny * Nz < 20 * 10**6:
            size_lim = [Nx, Ny, Nz]
        else:
            size_lim = [min(1024, Nx), min(1024, Ny), max(10, int(10**7 / (min(1024, Nx) * min(1024, Ny))))]

        verbmem = False
        paralpool = True
        clusters = 6

        # Parsing degli input opzionali
        if len(args) > 0:
            path_out = args[0]

        if len(args) > 1:
            sig_G = args[1]
            if not isinstance(sig_G, (int, float, list)):
                raise ValueError('Non numeric sigma value for variable sig_G')
            elif isinstance(sig_G, list) and len(sig_G) > 2:
                raise ValueError('Variable sig_G must be a 1 or 2 elements array')
            elif isinstance(sig_G, list) and len(sig_G) == 2 and sig_G[1] <= 0:
                raise ValueError('Second element in sig_G should be greater than 0')
            
        if len(args) > 2:
            size_lim = args[2]
            if not isinstance(size_lim, list):
                raise ValueError('size_lim must be a list')
            elif any(s < 0 for s in size_lim):
                raise ValueError('No negative values are allowed for the maximum slab size')
            size_lim[0] = min(Nx, size_lim[0])
            if abs(size_lim[0] - Nx) < 10:
                size_lim[0] = Nx
            size_lim[1] = min(Ny, size_lim[1])
            if abs(size_lim[1] - Ny) < 10:
                size_lim[1] = Ny
            size_lim[2] = min(Nz, size_lim[2])
            if abs(size_lim[2] - Nz) < 10:
                size_lim[2] = Nz

        if len(args) > 3:
            verbmem = args[3]
            if not isinstance(verbmem, bool):
                verbmem = bool(verbmem)

        if len(args) > 4:
            paralpool = args[4]
            if not isinstance(paralpool, bool):
                paralpool = bool(paralpool)

        if len(args) > 5:
            clusters = args[5]
            if not isinstance(clusters, int):
                clusters = int(clusters)

        # Creazione della directory di output
        try:
            os.makedirs(path_out, exist_ok=True)
        except OSError as e:
            raise OSError(f"Error creating directory: {e}")

        os.chdir(path_out)

    # Definizione dei limiti per i ritagli
    """
    Qui vengono definite le dimensioni dei ritagli dell'immagine. nxiL e nxeL rappresentano 
    gli indici di inizio e fine dei ritagli lungo l'asse x, mentre nyiL e nyeL fanno lo stesso 
    per l'asse y. th_back è una soglia per il background, calcolata come il 2% del valore 
    massimo possibile dei pixel (determinato dalla profondità di bit dell'immagine).
    """
    nxiL = list(range(0, Nx, size_lim[0]))
    nxeL = [min(Nx, i) for i in range(size_lim[0], Nx + size_lim[0], size_lim[0])]
    nyiL = list(range(0, Ny, size_lim[1]))
    nyeL = [min(Ny, i) for i in range(size_lim[1], Ny + size_lim[1], size_lim[1])]
    th_back = 0.02 * (2 ** bit)

    # parametri dei ritagli
    win = size_lim[2]  # Dimensione del ritaglio lungo l'asse z
    safe = 3  # Margine z
    safe_xy = 3  # Margine xy
    k_seq = list(range(0, Nz, win))  # Numero di fette dell'immagine iniziale lungo l'asse z

    # Inizio Segmentazione

    """
    Per ogni ritaglio, vengono definiti gli indici di inizio (nxi) e fine (nxe). 
    nxiS e nxeS sono gli indici di inizio e fine con i margini di sicurezza. 
    NxC, NyC e NzC sono le dimensioni dei ritagli lungo gli assi x, y e z.
    """

    for nxi,nxe in zip(nxiL, nxeL):
        # Definisci gli indici per il ritaglio
        nxiS = max(0, nxi - safe_xy)
        nxeS = min(Nx, nxe + safe_xy)
        NxC = nxe - nxi

        # Ciclo sull'asse y
        for nyi,nye in zip(nyiL, nyeL):
            # Definisci gli indici per il ritaglio
            nyiS = max(0, nyi - safe_xy)
            nyeS = min(Ny, nye + safe_xy)
            NyC = nye - nyi

            # Ciclo sull'asse z
            for k in k_seq:
                # Definisci gli indici per il ritaglio
                xinit = max(0, k - safe)
                xend = min(Nz, k + win + safe)
                NzC = min(Nz, k + win) - k

                # Controlla se il ritaglio è già stato elaborato, in tal caso, continua...
                
                # Visualizza messaggi di stato
                print(f'number of clusters is: {clusters}')
                print(f'crop is x({nxi}:{nxe}), y({nyi}:{nye})')
                print(f'starting with slice {k} over {Nz}, window set to {win}')

                # Leggi il ritaglio dell'immagine 
                # (i) per gestire immagini molto grandi;
                # (ii) prima definisci la dimensione effettiva del ritaglio, 
                
                nx = len(range(nxiS, nxeS))
                ny = len(range(nyiS, nyeS))
                nz = xend - xinit
                cIMc = np.zeros((nx, ny, nz), dtype=tp)

                # (iii) itera sulle fette dell'immagine originale per leggere il ritaglio

                for zz in range(cIMc.shape[2]):
                    cIMc[:, :, zz] = tifffile.imread(path_img, key=xinit + zz - 1)[nxiS:nxeS, nyiS:nyeS]

                # Filtro mediano se sig_G[0] == -1
                if sig_G[0] == -1:
                    print('applying 3x3x3 median filter...')
                    cIMc = median(cIMc, size=3)

                # Fornisce feedback all'utente
                print(f'1st of {len(sig_G)} passes...')

                # Derivate di primo ordine
                if sig_G[0] > 0:
                    cIMc_smoothed = gaussian(cIMc.astype(np.float32), sigma=[sig_G[0], sig_G[0] * res[0] / res[1], sig_G[0] * res[0] / res[2]])
                    Gx2, Gy2, Gz2 = np.gradient(cIMc_smoothed)
                else:
                    Gx2, Gy2, Gz2 = np.gradient(cIMc.astype(np.float32))

                # Derivate di secondo ordine
                Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = hessian_matrix(Gx2, sigma=1)
                Gxx2 = Hxx
                Gyy2 = Hyy
                Gzz2 = Hzz

                # Ritaglia i bordi (inclusi per evitare effetti di bordo nel calcolo delle derivate)

                # su cIMc
                cIMc = cIMc[
                min(safe_xy,nxi):min(NxC+min(safe_xy,nxi), cIMc.shape[0]),
                min(safe_xy,nxi):min(NxC+min(safe_xy,nxi), cIMc.shape[1]),
                min(safe_xy,k):min(win+min(safe_xy,k), cIMc.shape[2]),
                ]

                # su GxxKt
                GxxKt = Gxx2[
                    min(safe_xy,nxi):min(NxC+min(safe_xy,nxi), Gxx2.shape[0]),
                    min(safe_xy,nxi):min(NxC+min(safe_xy,nxi), Gxx2.shape[1]),
                    min(safe_xy,k):min(win+min(safe_xy,k), Gxx2.shape[2]),
                ]

                # su GyyKt
                GyyKt = Gyy2[
                    min(safe_xy,nxi):min(NxC+min(safe_xy,nxi), Gyy2.shape[0]),
                    min(safe_xy,nxi):min(NxC+min(safe_xy,nxi), Gyy2.shape[1]),
                    min(safe_xy,k):min(win+min(safe_xy,k), Gyy2.shape[2]),
                ]

                # su GzzKt
                GzzKt = Gzz2[
                    min(safe_xy,nxi):min(NxC+min(safe_xy,nxi), Gzz2.shape[0]),
                    min(safe_xy,nxi):min(NxC+min(safe_xy,nxi), Gzz2.shape[1]),
                    min(safe_xy,k):min(win+min(safe_xy,k), Gzz2.shape[2]),
                ]

                # Salva il ritaglio corrente dell'immagine
                with open(os.path.join(path_out, f'sl{k}_{nxi}_{nxe}_{nyi}_{nye}.pkl'), 'wb') as f:
                    pickle.dump({
                        'cIMc': cIMc,
                        'GxxKt': GxxKt,
                        'GyyKt': GyyKt,
                        'GzzKt': GzzKt
                    }, f)

                # Maschera il background a intensità quasi zero per ridurre l'uso della memoria
                mask_back = np.where(cIMc.ravel() >= th_back)[0]
                resiz_km = np.ones(len(cIMc.ravel()), dtype=np.uint8)

                # Definisci lo spazio delle caratteristiche per il k-means
                km_in1 = np.column_stack((
                    cIMc.ravel()[mask_back].astype(np.float32),
                    GxxKt.ravel()[mask_back],
                    GyyKt.ravel()[mask_back],
                    GzzKt.ravel()[mask_back]
                ))

                # Initialize TOT_KM1 with proper shape
                TOT_KM1 = np.ones((NxC, NyC, NzC), dtype=tp)

                # Clustering k-means
                kmeans = KMeans(n_clusters=clusters, n_init=10, max_iter=1000, random_state=42)
                labels = kmeans.fit_predict(km_in1.astype(np.float32))
                resiz_km[mask_back] = labels.astype(np.uint8)
                TOT_KM1 = resiz_km.reshape((NxC, NyC, NzC))
    

                # Save the k-means clustering results for the current crop
                with open(os.path.join(f'sl{k}_{nxi}_{nxe}_{nyi}_{nye}.pkl'), 'ab') as f:
                    pickle.dump({'TOT_KM1': TOT_KM1}, f)

                

def senpai_recompose(sig_G, Nx, Ny, Nz, size_lim, path_out, verbmem=False, tp=np.uint16):
    print('Composing segmentations...')
    def imfill_3d(image):
        """3D hole filling using scipy.ndimage"""
        from scipy import ndimage
        
        # Fill holes in each 2D slice
        filled = np.zeros_like(image, dtype=bool)
        for z in range(image.shape[2]):
            # Use binary fill for each slice
            filled[:,:,z] = ndimage.binary_fill_holes(image[:,:,z])
        
        # Optional: additional filling in other dimensions
        for y in range(image.shape[1]):
            filled[:,y,:] = ndimage.binary_fill_holes(filled[:,y,:])
        
        for x in range(image.shape[0]):
            filled[x,:,:] = ndimage.binary_fill_holes(filled[x,:,:])
        
        return filled
    
    def analyze_cluster_derivatives(GxxKt, GyyKt, GzzKt, cluster_labels, num_clusters):
        """
        Compute mean second-order derivatives for each cluster
        
        Parameters:
        - GxxKt, GyyKt, GzzKt: Second-order derivative matrices
        - cluster_labels: Cluster assignment for each pixel
        - num_clusters: Total number of clusters
        
        Returns:
        - Mean derivatives for each cluster
        """
        Gxx_cl = np.zeros(num_clusters)
        Gyy_cl = np.zeros(num_clusters)
        Gzz_cl = np.zeros(num_clusters)
        
        for clu in range(1, num_clusters + 1):
            cluster_mask = (cluster_labels == clu)
            Gxx_cl[clu-1] = np.mean(GxxKt[cluster_mask])
            Gyy_cl[clu-1] = np.mean(GyyKt[cluster_mask])
            Gzz_cl[clu-1] = np.mean(GzzKt[cluster_mask])
        
        # Find cluster with predominantly positive derivatives
        valid_cluster = np.where(~((Gxx_cl < 0) & (Gyy_cl < 0) & (Gzz_cl < 0)))[0][-1] + 1
        
        return valid_cluster
    
    def advanced_segmentation_refinement(initial_seg, original_image, bit_depth):
        """
        Refine segmentation by:
        - Filling holes
        - Applying intensity-based constraints
        - Cleaning small clusters
        """
        # Compute threshold based on bit depth
        intensity_threshold = 0.55 * (2 ** bit_depth)
        
        # 3D hole filling
        filled_seg = imfill_3d(initial_seg)
        
        # Additional segmentation refinement
        refined_seg = np.zeros_like(filled_seg, dtype=bool)
        
        # Process each slice and dimension
        for z in range(filled_seg.shape[2]):
            # Fill holes in 2D slice
            slice_seg = ndimage.binary_fill_holes(filled_seg[:,:,z])
            
            # Additional intensity-based segmentation
            slice_intensity = original_image[:,:,z] > intensity_threshold
            slice_combined = slice_seg | slice_intensity
            
            # Remove small artifacts
            slice_cleaned = ndimage.binary_opening(slice_combined)
            refined_seg[:,:,z] = slice_cleaned
        
        return refined_seg
    
    def process_3d_confocal_images(path_out, Nx, Ny, Nz, size_lim, win, sig_G, tp):
        """
        Full pipeline for processing 3D confocal images of neurons.
        """
        # Step 1: Recompose image from crops
        crop_results = recompose_crops(path_out, Nx, Ny, Nz, size_lim, win, sig_G, tp)
        senpai_KM_lv1 = crop_results['senpai_KM_lv1']
        senpai_KM_lv2 = crop_results['senpai_KM_lv2']
        cIM = crop_results['cIM']
        bitlr = crop_results['bitlr']
        
        # Step 2: Perform advanced segmentation refinement
        refined_seg = advanced_segmentation_refinement(
            initial_seg=senpai_KM_lv1,
            original_image=cIM,
            bit_depth=bitlr
        )
        
        # Step 3: Analyze cluster derivatives (if applicable)
        if senpai_KM_lv2 is not None:
            valid_cluster = analyze_cluster_derivatives(
                GxxKt, GyyKt, GzzKt,
                cluster_labels=senpai_KM_lv2,
                num_clusters=len(sig_G)
            )
            print(f"Valid cluster: {valid_cluster}")
        
        # Step 4: Return results
        return {
            'refined_segmentation': refined_seg,
            'cIM': cIM,
            'senpai_KM_lv1': senpai_KM_lv1,
            'senpai_KM_lv2': senpai_KM_lv2
        }
        
        # Placeholder for more complex segmentation logic
        # This would require translating the Matlab-specific image processing functions

        print(f'{im_in} : DONE.')
        
        return results
    
    def recompose_crops(path_out, Nx, Ny, Nz, size_lim, win, sig_G, tp):
        """
        Recompose 3D image from individual crops and prepare initial data.
        
        Parameters:
        - path_out: Directory containing crop files
        - Nx, Ny, Nz: Dimensions of the full 3D image
        - size_lim: Tuple specifying crop size limits in (x, y)
        - win: Number of slices per crop along the z-axis
        - sig_G: List of scales for derivative computations
        - tp: Data type of the original image
        
        Returns:
        - Dictionary containing recomposed full-size images:
            'senpai_KM_lv1': Initial segmentation level 1
            'senpai_KM_lv2': Initial segmentation level 2 (if sig_G > 1)
            'cIM': Reconstructed original image
            'bitlr': Bit depth of the original image
        """
        # Initialize full-size arrays
        senpai_KM_lv1 = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
        cIM = np.zeros((Nx, Ny, Nz), dtype=tp)
        
        if len(sig_G) > 1:
            senpai_KM_lv2 = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
        else:
            senpai_KM_lv2 = None
        
        # Iterate through image crops
        for k in range(0, Nz, win):
            for x in range(0, Nx, size_lim[0]):
                for y in range(0, Ny, size_lim[1]):
                    # Construct crop filename
                    crop_filename = f'sl{k}_{x}_{min(x+size_lim[0]-1, Nx)}_{y}_{min(y+size_lim[1]-1, Ny)}.pkl'
                    
                    # Load crop data
                    with open(os.path.join(path_out, crop_filename), 'rb') as f:
                        crop_data = pickle.load(f)
                    
                    # Define crop boundaries
                    x_end = min(x + size_lim[0], Nx)
                    y_end = min(y + size_lim[1], Ny)
                    z_end = min(k + win, Nz)
                    
                    # Recompose crops
                    senpai_KM_lv1[x:x_end, y:y_end, k:z_end] = crop_data['TOT_KM1']
                    cIM[x:x_end, y:y_end, k:z_end] = crop_data['cIMc']
                    
                    if senpai_KM_lv2 is not None:
                        senpai_KM_lv2[x:x_end, y:y_end, k:z_end] = crop_data['TOT_KM2']
        
        # Compute bit depth for intensity thresholding
        bitlr = int(np.ceil(np.log2(float(np.max(cIM)))))
        
        # Return recomposed images and bit depth
        return {
            'senpai_KM_lv1': senpai_KM_lv1,
            'senpai_KM_lv2': senpai_KM_lv2,
            'cIM': cIM,
            'bitlr': bitlr
        }


# Parametri di input (esempio)
size_lim = [256, 256, 149]
clusters = 6 # Numero di cluster per il k-means (default = 6)
path_img = 'test_data.tif'

