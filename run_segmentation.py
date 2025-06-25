import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tifffile import imread
import os

# Import our NeuronSegmentation class (assuming it's in a file called neuron_segmentation.py)
from neuron_segmentation import NeuronSegmenter, SegmentationConfig

def process_neuron_image(image_path: str, output_path: str):
    """
    Process a 3D confocal image of neurons using the NeuronSegmentation class
    
    Args:
        image_path: Path to the input image file
        output_path: Path where results will be saved
    """
    # 1. Set up configuration
    config = SegmentationConfig(
        size_lim=[256, 256, 64],  # Crop size for processing
        clusters=6,                # Number of clusters for k-means
        sigma_gaussian=[0],     # Gaussian smoothing parameters
        safe_margin=3,            # Overlap margin for crops
        safe_margin_xy=3,         # Overlap in xy plane
        background_threshold=None  # Will be automatically set based on bit depth
    )
    
    # 2. Initialize the segmentation class
    segmenter = NeuronSegmentation(config)
    
    # 3. Load the image
    image_dir = os.path.dirname(image_path)
    image_name = os.path.basename(image_path)
    segmenter.load_image(image_dir, image_name)
    
    # 4. Get image dimensions
    Nx, Ny, Nz = segmenter.image_info['dimensions']
    
    # 5. Process the image in crops
    results = {
        'segmentation': np.zeros((Nx, Ny, Nz), dtype=np.uint8),
        'derivatives': {
            'Gxx': np.zeros((Nx, Ny, Nz), dtype=np.float32),
            'Gyy': np.zeros((Nx, Ny, Nz), dtype=np.float32),
            'Gzz': np.zeros((Nx, Ny, Nz), dtype=np.float32)
        }
    }
    
    # Calculate crop parameters
    crop_size_x, crop_size_y, crop_size_z = config.size_lim
    
    for z in range(0, Nz, crop_size_z):
        for y in range(0, Ny, crop_size_y):
            for x in range(0, Nx, crop_size_x):
                # Define crop boundaries
                x_end = min(x + crop_size_x, Nx)
                y_end = min(y + crop_size_y, Ny)
                z_end = min(z + crop_size_z, Nz)
                
                # Load crop with safety margins
                x_start_safe = max(0, x - config.safe_margin_xy)
                y_start_safe = max(0, y - config.safe_margin_xy)
                z_start_safe = max(0, z - config.safe_margin)
                x_end_safe = min(Nx, x_end + config.safe_margin_xy)
                y_end_safe = min(Ny, y_end + config.safe_margin_xy)
                z_end_safe = min(Nz, z_end + config.safe_margin)
                
                # Read image crop
                crop = imread(image_path)[z_start_safe:z_end_safe, 
                                        y_start_safe:y_end_safe, 
                                        x_start_safe:x_end_safe]
                
                # Process crop
                crop_info = {
                    'x_range': (x, x_end),
                    'y_range': (y, y_end),
                    'z_range': (z, z_end),
                    'safe_margins': (config.safe_margin_xy, config.safe_margin)
                }
                
                processed_crop = segmenter.process_crop(crop, crop_info)
                
                # Remove safety margins from results
                margin_x = config.safe_margin_xy
                margin_y = config.safe_margin_xy
                margin_z = config.safe_margin
                
                crop_segmentation = processed_crop['clustering']
                crop_derivatives = processed_crop['derivatives']
                
                # Store results without safety margins
                results['segmentation'][x:x_end, y:y_end, z:z_end] = \
                    crop_segmentation[margin_x:margin_x + (x_end - x),
                                    margin_y:margin_y + (y_end - y),
                                    margin_z:margin_z + (z_end - z)]
                
                for deriv_key in ['Gxx', 'Gyy', 'Gzz']:
                    results['derivatives'][deriv_key][x:x_end, y:y_end, z:z_end] = \
                        crop_derivatives[deriv_key][margin_x:margin_x + (x_end - x),
                                                  margin_y:margin_y + (y_end - y),
                                                  margin_z:margin_z + (z_end - z)]
                
                print(f"Processed crop: x={x}-{x_end}, y={y}-{y_end}, z={z}-{z_end}")
    
    # 6. Refine the segmentation
    refined_segmentation = segmenter.refine_segmentation(
        results['segmentation'],
        imread(image_path)
    )
    results['refined_segmentation'] = refined_segmentation
    
    # 7. Save results
    segmenter.save_results(results, output_path)
    
    # 8. Visualize some results (middle z-slice)
    mid_z = Nz // 2
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    # Original image
    axes[0].imshow(imread(image_path)[:, :, mid_z], cmap='gray')
    axes[0].set_title('Original Image')
    
    # Initial segmentation
    axes[1].imshow(results['segmentation'][:, :, mid_z], cmap='nipy_spectral')
    axes[1].set_title('Initial Segmentation')
    
    # Refined segmentation
    axes[2].imshow(results['refined_segmentation'][:, :, mid_z], cmap='nipy_spectral')
    axes[2].set_title('Refined Segmentation')
    
    # Derivatives (Gxx)
    axes[3].imshow(results['derivatives']['Gxx'][:, :, mid_z], cmap='coolwarm')
    axes[3].set_title('Second Derivative (Gxx)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'segmentation_results.png'))
    plt.close()

# Example usage
if __name__ == "__main__":
    # Define paths
    input_image = "test_data.tif"
    result_dir = "results"
    out_dir = "test_output"
    output_directory = os.path.join(result_dir, out_dir)
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Process the image
    process_neuron_image(input_image, output_directory)