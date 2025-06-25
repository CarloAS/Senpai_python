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



def senpai_seg_core_v4(path_in, im_in, *args):
    """
    Segments a 3D .tiff image using a topology-informed k-means-based clustering approach.
    
    Parameters:
        path_in (str): Path where the input file is located.
        im_in (str): Filename of the input image (8 or 16-bit .tiff).
        *args: Optional parameters in the following order:
            - path_out (str): Path to save the outputs. Default: './senpaiseg/'.
            - sig_G (list or float): Gaussian kernel sigma(s). Default: 0.
            - size_lim (list): Maximum crop size to handle memory constraints. Default: calculated based on input size.
            - verbmem (bool): Whether to save partial results. Default: False.
            - paralpool (bool): Whether to enable parallel processing. Default: True.
            - clusters (int): Number of k-means clusters. Default: 6.
    
    Raises:
        ValueError: If required arguments are missing or invalid values are provided.
        FileNotFoundError: If the input image file is not found.
    """
    # Handle input arguments
    if path_in is None or im_in is None:
        raise ValueError("Input arguments 'path_in' and 'im_in' are required!")
    
    input_file = os.path.join(path_in, im_in)
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
    
    print(f"Input file: {input_file}")
    
    # Read image metadata
    try:
        with tifffile.TiffFile(input_file) as tif:
            metadata = tif.pages[0].tags
            Ny, Nx = tif.pages[0].shape
            Nz = len(tif.pages)
            bit_depth = metadata['BitsPerSample'].value
            xres, yres = metadata.get('XResolution', 1), metadata.get('YResolution', 1)
            zres = 1  # Default voxel size

            # Extract z-resolution if available
            description = metadata.get('ImageDescription', None)
            if description:
                description = description.value.decode('utf-8')
                if 'PhysicalSizeZ=' in description:
                    zres = float(description.split('PhysicalSizeZ=')[1].split()[0])
                elif 'spacing=' in description:
                    zres = float(description.split('spacing=')[1].split()[0])

    except Exception as e:
        raise ValueError(f"Error reading image metadata: {e}")
    
    # Validate bit depth
    if bit_depth == 16:
        dtype = np.float32
    elif bit_depth == 8:
        dtype = np.uint8
    else:
        raise ValueError("Input image must be 8 or 16-bit.")
    
    # Defaults
    path_out = args[0] if len(args) > 0 else os.path.join(os.getcwd(), 'senpaiseg')
    sig_G = args[1] if len(args) > 1 else [0]
    if not isinstance(sig_G, (list, tuple, int, float)):
        raise ValueError("Non-numeric sigma value for 'sig_G'.")
    if isinstance(sig_G, list) and len(sig_G) > 2:
        raise ValueError("'sig_G' must be a 1- or 2-element list.")
    if isinstance(sig_G, list) and len(sig_G) == 2 and sig_G[1] <= 0:
        raise ValueError("Second element in 'sig_G' must be greater than 0.")

    if Nx * Ny * Nz < 20 * 10**6:
        size_lim = [Nx, Ny, Nz]
    else:
        size_lim = [min(1024, Nx), min(1024, Ny), max(10, 10**7 // (min(1024, Nx) * min(1024, Ny)))]
    if len(args) > 2:
        size_lim = args[2]
        if any(s < 0 for s in size_lim):
            raise ValueError("Negative values are not allowed for 'size_lim'.")
        size_lim = [min(Nx, size_lim[0]), min(Ny, size_lim[1]), min(Nz, size_lim[2])]

    verbmem = args[3] if len(args) > 3 else False
    if not isinstance(verbmem, bool):
        verbmem = bool(verbmem)

    paralpool = args[4] if len(args) > 4 else True
    if not isinstance(paralpool, bool):
        paralpool = bool(paralpool)

    clusters = args[5] if len(args) > 5 else 6
    if not isinstance(clusters, int):
        clusters = int(clusters)

    # Create output directory
    os.makedirs(path_out, exist_ok=True)
    os.chdir(path_out)
    print(f"Output directory: {path_out}")

    return {
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz,
        "bit_depth": bit_depth,
        "dtype": dtype,
        "xres": xres,
        "yres": yres,
        "zres": zres,
        "size_lim": size_lim,
        "verbmem": verbmem,
        "paralpool": paralpool,
        "clusters": clusters,
        "path_out": path_out
    }

def senpai_segmentation(image_path, output_path, size_lim, clusters, sig_G, xres, yres, zres, bitl=8, verbose=True):
    """
    Core segmentation function for large 3D images.
    
    Parameters:
        image_path (str): Path to the input image stack.
        output_path (str): Directory to save segmentation results.
        size_lim (tuple): (crop_x, crop_y, crop_z) defines the crop size.
        clusters (int): Number of clusters for k-means.
        sig_G (list): Gaussian filter standard deviations for each pass.
        xres, yres, zres (float): Voxel dimensions.
        bitl (int): Bit depth of the image, default 8.
        verbose (bool): If True, prints progress and debug information.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load the image dimensions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image_dims = io.imread(image_path).shape
    
    Nx, Ny, Nz = image_dims[1], image_dims[2], image_dims[0]
    th_back = 0.02 * (2 ** bitl)
    win = size_lim[2]
    safe = 3
    safe_xy = 3
    k_seq = range(1, Nz, win)
    
    # Define crop ranges
    nxiL = range(1, Nx, size_lim[0])
    nxeL = [min(Nx, i + size_lim[0] - 1) for i in nxiL]
    nyiL = range(1, Ny, size_lim[1])
    nyeL = [min(Ny, i + size_lim[1] - 1) for i in nyiL]
    
    # Start segmentation
    for nxi, nxe in zip(nxiL, nxeL):
        nxiS = max(1, nxi - safe_xy)
        nxeS = min(Nx, nxe + safe_xy)
        NxC = nxe - nxi + 1
        
        for nyi, nye in zip(nyiL, nyeL):
            nyiS = max(1, nyi - safe_xy)
            nyeS = min(Ny, nye + safe_xy)
            NyC = nye - nyi + 1
            
            for k in k_seq:
                xinit = max(1, k - safe)
                xend = min(Nz, k + win + safe)
                NzC = min(Nz, k + win - 1) - k + 1
                
                crop_file = os.path.join(output_path, f'sl{k}_{nxi}_{nxe}_{nyi}_{nye}.npz')
                if os.path.exists(crop_file):
                    if verbose:
                        print(f"Crop already processed: {crop_file}")
                    continue
                
                # Load and crop the image
                cIMc = np.zeros((nxeS - nxiS + 1, nyeS - nyiS + 1, xend - xinit + 1), dtype=np.float32)
                for z in range(cIMc.shape[2]):
                    cIMc[:, :, z] = io.imread(image_path, key=xinit + z - 1)[nxiS:nxeS + 1, nyiS:nyeS + 1]
                
                # Apply median filter if specified
                if sig_G[0] == -1:
                    cIMc = ndi.median_filter(cIMc, size=(3, 3, 3))
                
                # Compute first-order derivatives
                if sig_G[0] > 0:
                    cIMc_smoothed = gaussian(cIMc, sigma=(sig_G[0], sig_G[0] * xres / yres, sig_G[0] * xres / zres))
                    Gx, Gy, Gz = np.gradient(cIMc_smoothed)
                else:
                    Gx, Gy, Gz = np.gradient(cIMc)
                
                # Compute second-order derivatives
                Gxx = np.gradient(Gx, axis=0)
                Gyy = np.gradient(Gy, axis=1)
                Gzz = np.gradient(Gz, axis=2)
                
                # Mask background
                mask_back = cIMc >= th_back
                km_in = np.stack([cIMc[mask_back], Gxx[mask_back], Gyy[mask_back], Gzz[mask_back]], axis=1)
                
                # K-means clustering
                if km_in.shape[0] >= 10:
                    kmeans = KMeans(n_clusters=clusters, n_init=10, max_iter=1000, random_state=42)
                    TOT_KM1 = kmeans.fit_predict(km_in)
                else:
                    TOT_KM1 = np.ones(cIMc.shape, dtype=np.uint8)
                
                # Save results
                np.savez_compressed(crop_file, cIMc=cIMc, Gxx=Gxx, Gyy=Gyy, Gzz=Gzz, TOT_KM1=TOT_KM1)
                
                if verbose:
                    print(f"Processed crop: x({nxi}:{nxe}), y({nyi}:{nye}), slice {k}/{Nz}")
    
    if verbose:
        print("Recomposing crops...")
    
    senpai_recompose(sig_G, Nx, Ny, Nz, size_lim, output_path)

def senpai_recompose(sig_G, Nx, Ny, Nz, size_lim, output_path, win, tp):
    """
    This function recomposes processed slabs into a single matrix of
    k-means clusters having the same size as the original image.

    Parameters:
        sig_G (list): Gaussian kernel standard deviations for smoothing.
        Nx, Ny, Nz (int): Dimensions of the original image.
        size_lim (tuple): Crop size limits (x, y, z).
        output_path (str): Path to save the recomposed output.
        win (int): Window size for z-slices.
        tp (int): Number of image channels.

    Returns:
        None
    """
    print("Composing segmentations...")
    
    # Initialize recomposed matrices
    senpai_KM_lv1 = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
    cIM = np.zeros((Nx, Ny, Nz, tp), dtype=np.float32)

    # For multiple runs
    senpai_KM_lv2 = None
    if len(sig_G) > 1:
        senpai_KM_lv2 = np.zeros((Nx, Ny, Nz), dtype=np.uint8)
    
    # Loop through z-slices
    for k in range(1, Nz + 1, win):
        for x in range(1, Nx + 1, size_lim[0]):
            for y in range(1, Ny + 1, size_lim[1]):
                # Compute crop boundaries
                x_end = min(x + size_lim[0] - 1, Nx)
                y_end = min(y + size_lim[1] - 1, Ny)
                k_end = min(k + win - 1, Nz)
                
                # Load the processed crop
                crop_file = f"sl{k}_{x}_{x_end}_{y}_{y_end}.mat"
                crop_path = os.path.join(output_path, crop_file)
                if not os.path.isfile(crop_path):
                    print(f"Warning: File {crop_path} not found. Skipping.")
                    continue
                tmpcrp = sio.loadmat(crop_path)
                
                # Recompose level 1 segmentation
                senpai_KM_lv1[x - 1:x_end, y - 1:y_end, k - 1:k_end] = tmpcrp['TOT_KM1']
                cIM[x - 1:x_end, y - 1:y_end, k - 1:k_end] = tmpcrp['cIMc']

                # Recompose level 2 segmentation if applicable
                if len(sig_G) > 1 and 'TOT_KM2' in tmpcrp:
                    senpai_KM_lv2[x - 1:x_end, y - 1:y_end, k - 1:k_end] = tmpcrp['TOT_KM2']

    # Calculate bit depth
    bitlr = np.ceil(np.log2(np.max(cIM))).astype(int)

    # Save the recomposed data
    final_output = {'senpai_KM_lv1': senpai_KM_lv1, 'cIM': cIM}
    if len(sig_G) > 1 and senpai_KM_lv2 is not None:
        final_output['senpai_KM_lv2'] = senpai_KM_lv2

    save_path = os.path.join(output_path, 'senpai_final.mat')
    sio.savemat(save_path, final_output)
    print(f"Recomposed segmentations saved to {save_path}")

def senpai_define_classes(sig_G, Nx, Ny, Nz, size_lim, output_path, win, tp, clusters, verbmem, cIM, bitlr):
    """
    Defines classes for neural structures and recomposes segmentations.
    Saves updated masks and final segmentations.

    Parameters:
        sig_G (list): Gaussian kernel standard deviations for smoothing.
        Nx, Ny, Nz (int): Dimensions of the original image.
        size_lim (tuple): Crop size limits (x, y, z).
        output_path (str): Path to save the recomposed output.
        win (int): Window size for z-slices.
        tp (int): Number of image channels.
        clusters (int): Number of k-means clusters.
        verbmem (bool): Whether to delete temporary files after processing.
        cIM (np.ndarray): Input intensity matrix.
        bitlr (int): Bit depth of the intensity matrix.

    Returns:
        None
    """
    print("Defining classes for neural structures...")

    # Initialize the masks
    senpai_KM_lv1_msk = np.zeros((Nx, Ny, Nz), dtype=np.uint8)

    # Level 1: Compute mean values for second-order derivatives within each cluster
    for k in range(1, Nz + 1, win):
        for x in range(1, Nx + 1, size_lim[0]):
            for y in range(1, Ny + 1, size_lim[1]):
                # Compute crop boundaries
                x_end = min(x + size_lim[0] - 1, Nx)
                y_end = min(y + size_lim[1] - 1, Ny)
                k_end = min(k + win - 1, Nz)

                # Load the crop
                crop_file = f"sl{k}_{x}_{x_end}_{y}_{y_end}.mat"
                crop_path = os.path.join(output_path, crop_file)
                if not os.path.isfile(crop_path):
                    print(f"Warning: File {crop_path} not found. Skipping.")
                    continue

                tmpcrp = sio.loadmat(crop_path)
                GxxKt = tmpcrp['GxxKt']
                GyyKt = tmpcrp['GyyKt']
                GzzKt = tmpcrp['GzzKt']

                # Calculate mean values for each cluster
                Gxx_cl = np.zeros(clusters)
                Gyy_cl = np.zeros(clusters)
                Gzz_cl = np.zeros(clusters)

                for clu in range(1, clusters + 1):
                    mask = senpai_KM_lv1_msk[x - 1:x_end, y - 1:y_end, k - 1:k_end] == clu
                    Gxx_cl[clu - 1] = np.mean(GxxKt[mask])
                    Gyy_cl[clu - 1] = np.mean(GyyKt[mask])
                    Gzz_cl[clu - 1] = np.mean(GzzKt[mask])

                # Assign classes based on conditions
                condition = (Gxx_cl < 0) & (Gyy_cl < 0) & (Gzz_cl < 0)
                senpai_KM_lv1_msk[x - 1:x_end, y - 1:y_end, k - 1:k_end] = (
                    np.where(~condition)[0][-1] + 2
                )

    # Save updated Level 1 data
    save_path = os.path.join(output_path, "senpai_final.mat")
    sio.savemat(
        save_path, {"senpai_KM_lv1_msk": senpai_KM_lv1_msk, "Gxx_cl": Gxx_cl, "Gyy_cl": Gyy_cl, "Gzz_cl": Gzz_cl}, appendmat=True
    )

    # Level 2 processing if multiple Gaussian kernels
    if len(sig_G) > 1:
        senpai_KM_lv2_msk = np.zeros((Nx, Ny, Nz), dtype=np.uint8)

        for k in range(1, Nz + 1, win):
            for x in range(1, Nx + 1, size_lim[0]):
                for y in range(1, Ny + 1, size_lim[1]):
                    # Compute crop boundaries
                    x_end = min(x + size_lim[0] - 1, Nx)
                    y_end = min(y + size_lim[1] - 1, Ny)
                    k_end = min(k + win - 1, Nz)

                    # Load the crop
                    crop_file = f"sl{k}_{x}_{x_end}_{y}_{y_end}.mat"
                    crop_path = os.path.join(output_path, crop_file)
                    if not os.path.isfile(crop_path):
                        print(f"Warning: File {crop_path} not found. Skipping.")
                        continue

                    tmpcrp = sio.loadmat(crop_path)
                    GxxKt2 = tmpcrp['GxxKt2']
                    GyyKt2 = tmpcrp['GyyKt2']
                    GzzKt2 = tmpcrp['GzzKt2']

                    # Calculate mean values for each cluster
                    Gxx_cl = np.zeros(clusters)
                    Gyy_cl = np.zeros(clusters)
                    Gzz_cl = np.zeros(clusters)

                    for clu in range(1, clusters + 1):
                        mask = senpai_KM_lv2_msk[x - 1:x_end, y - 1:y_end, k - 1:k_end] == clu
                        Gxx_cl[clu - 1] = np.mean(GxxKt2[mask])
                        Gyy_cl[clu - 1] = np.mean(GyyKt2[mask])
                        Gzz_cl[clu - 1] = np.mean(GzzKt2[mask])

                    # Assign classes based on conditions
                    condition = (Gxx_cl < 0) & (Gyy_cl < 0) & (Gzz_cl < 0)
                    senpai_KM_lv2_msk[x - 1:x_end, y - 1:y_end, k - 1:k_end] = (
                        np.where(~condition)[0][-1] + 2
                    )

        # Save updated Level 2 data
        sio.savemat(
            save_path,
            {"senpai_KM_lv2_msk": senpai_KM_lv2_msk, "Gxx_cl": Gxx_cl, "Gyy_cl": Gyy_cl, "Gzz_cl": Gzz_cl},
            appendmat=True,
        )

    # Optional: Delete temporary files
    if not verbmem:
        temp_files = [f for f in os.listdir(output_path) if f.startswith("sl")]
        for temp_file in temp_files:
            os.remove(os.path.join(output_path, temp_file))

    print("Segmentation completed.")

def senpai_separator(senpai_seg, cIM, somas):
    """
    Separates neurons from a k-means segmentation.

    Parameters:
        senpai_seg (np.ndarray): Logical matrix with the final segmentation produced by senpai_seg_core.
        cIM (np.ndarray): Numeric matrix of the image that generated the segmentation.
        somas (np.ndarray): Logical matrix encoding a binary segmentation of the somas in the image.

    Returns:
        parcel_final (np.ndarray): Numeric matrix with the parcellation of the final segmentation.
    """
    print("Separating neurons...")

    # Ensure `senpai_seg` is a logical matrix
    senpai_seg = senpai_seg.astype(bool)

    # Take the negative of the median-filtered image
    db = np.max(cIM)
    cIM_inv = db - ndi.median_filter(cIM, size=3)
    cIM_inv[~senpai_seg] = db
    del cIM

    # Impose minima in the mask of somas
    cIM_inv = np.where(somas, 0, cIM_inv)

    # Watershed transform
    ww = watershed(cIM_inv).astype(np.uint16)

    # Provide final parcellation
    parcel_final = ww * senpai_seg.astype(np.uint16)

    # Save intermediate result
    sio.savemat("senpai_separator.mat", {"ww": ww})

    # Remove non-connected pieces
    print("Pruning non-connected branches...")
    for vv in range(1, np.max(ww) + 1):
        # Find connected components
        mask = parcel_final == vv
        labeled, num_features = label(mask, connectivity=3, return_num=True)
        regions = regionprops(labeled)

        # Keep only the largest connected component
        if regions:
            largest_region = max(regions, key=lambda r: r.area)
            for region in regions:
                if region != largest_region:
                    coords = region.coords
                    parcel_final[coords[:, 0], coords[:, 1], coords[:, 2]] = 0

    # Save final parcellation
    sio.savemat("senpai_separator.mat", {"parcel_final": parcel_final}, appendmat=True)

    print("Done!")
    return parcel_final

def senpai_skeletonize(cIM, neuron, somas, ss, folder_name):
    """
    Produces a skeleton of a binary segmentation and outputs a graph tree structure 
    and SWC-formatted matrix.

    Parameters:
        cIM (np.ndarray): Original image stack on which the segmentation has been produced.
        neuron (np.ndarray): Binary segmentation of a single neuron.
        somas (np.ndarray): Binary mask of one (or multiple) somas.
        ss (int): Skeleton identifier for output file naming.
        folder_name (str): Directory where results will be saved.

    Returns:
        t (networkx.Graph): Minimum spanning tree of the neuron.
        swc (np.ndarray): SWC-like matrix encoding the skeleton.
    """
    print("Starting skeletonization...")

    somas = somas > 0

    # Include soma mask in the segmentation
    neuron = (neuron > 0) | somas

    # Keep only the largest connected component (neuron + soma)
    labeled_neuron, num_features = label(neuron, connectivity=3, return_num=True)
    region_sizes = np.array([np.sum(labeled_neuron == i) for i in range(1, num_features + 1)])
    largest_region = np.argmax(region_sizes) + 1
    neuron = (labeled_neuron == largest_region).astype(np.uint8)

    # Fill holes
    neuron = np.pad(neuron, 1, mode="constant")
    neuron = skeletonize_3d(neuron)
    neuron = neuron[1:-1, 1:-1, 1:-1]

    # Refine somas mask to match neuron region
    somas = somas * neuron

    # Distance transform
    bwd = ndi.distance_transform_edt(neuron)

    # Skeletonize the neuron
    skeleton = skeletonize_3d(neuron)

    # Extract skeleton voxel indices
    points = np.argwhere(skeleton)
    xn, yn, zn = points[:, 0], points[:, 1], points[:, 2]

    # Build the graph from the skeleton
    print("Building graph from skeleton...")
    G = nx.Graph()
    done = set()

    while len(done) < len(points):
        seed_idx = next(idx for idx in range(len(points)) if idx not in done)
        seed_point = points[seed_idx]

        # Find neighbors within a radius of sqrt(3)
        distances = np.sum((points - seed_point) ** 2, axis=1)
        neighbors = np.where(distances <= 3)[0]
        neighbors = [n for n in neighbors if n != seed_idx and n not in done]

        # Add edges to the graph
        for neighbor in neighbors:
            weight = cIM[tuple(points[neighbor])]
            G.add_edge(seed_idx, neighbor, weight=weight)

        done.add(seed_idx)

    # Convert graph to a minimum spanning tree
    print("Building minimum spanning tree...")
    mst = nx.minimum_spanning_tree(G)

    # Ensure the graph is connected
    components = list(nx.connected_components(mst))
    if len(components) > 1:
        # Find leaves near the soma
        soma_coords = np.argwhere(somas)
        soma_center = np.mean(soma_coords, axis=0).astype(int)

        # Add soma node
        soma_idx = len(points)
        points = np.vstack((points, soma_center))
        mst.add_node(soma_idx)

        for component in components:
            # Connect the closest point in the component to the soma
            component_points = [points[i] for i in component]
            distances = np.linalg.norm(component_points - soma_center, axis=1)
            closest_point_idx = component[np.argmin(distances)]
            mst.add_edge(soma_idx, closest_point_idx, weight=cIM[tuple(points[closest_point_idx])])

    # Create SWC-like matrix
    print("Building SWC-like matrix...")
    pred = {node: None for node in mst.nodes}
    for edge in mst.edges:
        u, v = edge
        if pred[v] is None:
            pred[v] = u
        elif pred[u] is None:
            pred[u] = v

    swc = []
    for i, (x, y, z) in enumerate(points):
        node_type = 0
        radius = bwd[x, y, z]
        parent = pred[i] if pred[i] is not None else -1
        swc.append([i + 1, node_type, x, y, z, radius, parent + 1])

    swc = np.array(swc)

    # Save SWC file
    filename = os.path.join(folder_name, f"neuron_skel_{ss}.txt")
    np.savetxt(filename, swc, fmt="%.6f", delimiter=" ")

    # Rename file to .swc
    swc_filename = os.path.splitext(filename)[0] + ".swc"
    os.rename(filename, swc_filename)

    print("Skeletonization complete.")
    return mst, swc

def senpai_prune(parcellation, *args):
    """
    This function displays a GUI to assess and mark neural branches for pruning.
    Args:
        parcellation: numpy 3D array (uint8/uint16) containing a parcellation.
        nn (optional): Integer index of the first neuron to display (default=1).
        markers (optional): numpy 3D logical array for neural core masks.
    """
    N = np.max(parcellation)
    colormap = plt.cm.get_cmap('viridis', 101)
    nn = args[0] if len(args) > 0 else 1
    neuron = (parcellation == nn)
    markers = args[1] if len(args) > 1 else np.zeros_like(parcellation, dtype=bool)
    marker_regions = label(markers, connectivity=2)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
    ax3d = fig.add_subplot(111, projection='3d')

    def plot_neuron():
        nonlocal ax3d
        ax3d.clear()
        verts, faces, _, _ = marching_cubes(neuron, level=0.5)
        colors = colormap((verts[:, 2] - verts[:, 2].min()) / (verts[:, 2].ptp() + 1e-6))
        mesh = Poly3DCollection(verts[faces], facecolors=colors, edgecolors='none', linewidth=0.2)
        ax3d.add_collection3d(mesh)
        ax3d.set_box_aspect([1, 1, 1])
        ax3d.set_xlim(0, neuron.shape[0])
        ax3d.set_ylim(0, neuron.shape[1])
        ax3d.set_zlim(0, neuron.shape[2])
        ax3d.view_init(30, 120)
        ax3d.axis("off")
        plt.draw()

    def update_view(new_nn):
        nonlocal neuron, nn
        nn = np.clip(new_nn, 1, N)
        neuron[:] = (parcellation == nn)
        neuron_label.set_text(f"Currently visualizing neuron {nn} of {N}")
        plot_neuron()

    def save_markers(event):
        with open("markers.pkl", "wb") as f:
            pickle.dump(markers, f)
        print("Markers saved to markers.pkl")

    def next_neuron(event):
        update_view(nn + 1)

    def prev_neuron(event):
        update_view(nn - 1)

    def goto_neuron(event):
        try:
            target_nn = int(neuron_index.text)
            update_view(target_nn)
        except ValueError:
            print("Invalid neuron index")

    def mark_branch(event):
        cursor = plt.ginput(1)
        if cursor:
            x, y, z = map(int, cursor[0])
            markers[x, y, z] = True
            print(f"Marked branch at: {x}, {y}, {z}")
            plot_neuron()

    # GUI elements
    neuron_label = plt.text(0.5, -0.1, f"Currently visualizing neuron {nn} of {N}", transform=ax.transAxes, ha='center')
    save_button = Button(plt.axes([0.75, 0.05, 0.15, 0.075]), 'Save')
    save_button.on_clicked(save_markers)
    next_button = Button(plt.axes([0.55, 0.05, 0.1, 0.075]), '>')
    next_button.on_clicked(next_neuron)
    prev_button = Button(plt.axes([0.35, 0.05, 0.1, 0.075]), '<')
    prev_button.on_clicked(prev_neuron)
    neuron_index = TextBox(plt.axes([0.4, 0.01, 0.1, 0.05]), 'Neuron #')
    go_button = Button(plt.axes([0.55, 0.01, 0.1, 0.05]), 'Go!')
    go_button.on_clicked(goto_neuron)
    mark_button = Button(plt.axes([0.75, 0.15, 0.15, 0.075]), 'Mark Branch')
    mark_button.on_clicked(mark_branch)

    update_view(nn)
    plt.show()