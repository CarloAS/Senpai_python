"""
NeuronalSomaMarker - A Python module for marking neuronal somas in 3D image stacks.

This module provides tools for interactive marking of somas in 3D neuronal images.
It's designed to be both a standalone tool and a component that can be integrated
into larger neuronal analysis pipelines.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import tifffile
import json
import os
from scipy import ndimage
import argparse
from pathlib import Path
import matplotlib.cm as cm
import pandas as pd


class NeuronalSomaMarker:
    """
    A GUI tool for marking neuronal somas in 3D image stacks.
    This is a Python translation of the MATLAB senpai_somamark function.
    
    Attributes:
        image_path (str): Path to the image file
        image (ndarray): The loaded image data
        somas (list): List of soma locations and properties
        radius (int): Default radius for marking somas
        z_factor (int): Z-axis scaling factor
    """
    
    def __init__(self, image_data=None, image_path=None, output_path=None, 
                 init_marks=None, radius=10, z_factor=3):
        """
        Initialize the NeuronalSomaMarker.
        
        Parameters:
        -----------
        image_data : ndarray, optional
            3D or 4D image array, if provided directly instead of path
        image_path : str, optional
            Path to the 3D TIFF image stack (not needed if image_data is provided)
        output_path : str, optional
            Path to save the output JSON file
        init_marks : ndarray or str, optional
            Initial soma markers (array or path to JSON file)
        radius : int, optional
            Radius of the sphere used to mark somas
        z_factor : int, optional
            Z-axis scaling factor for the sphere
        """
        # Store parameters
        self.image_path = image_path
        self.radius = radius
        self.z_factor = z_factor
        self.somas = []  # Will store [x, y, z, radius, z_factor] for each soma
        self.fig = None
        self.ax = None
        self.coords = None
        
        # Handle output path
        if image_path and not output_path:
            self.output_path = os.path.splitext(image_path)[0] + '_somas.json'
        else:
            self.output_path = output_path or 'soma_markers.json'
        
        # Load the image
        if image_data is not None:
            self.image = image_data
            # Make sure it has the right dimensions
            if len(self.image.shape) == 3:  # Add channel dimension if grayscale
                self.image = self.image[..., np.newaxis]
        elif image_path:
            try:
                self.image = tifffile.imread(image_path)
                if len(self.image.shape) == 3:  # Add channel dimension if grayscale
                    self.image = self.image[..., np.newaxis]
            except Exception as e:
                raise ValueError(f"Error loading image from {image_path}: {str(e)}")
        else:
            raise ValueError("Either image_data or image_path must be provided")
        
        # Initialize slice to middle of stack
        self.slice_idx = self.image.shape[0] // 2
        
        # If initial marks are provided
        if init_marks is not None:
            self.load_initial_marks(init_marks)
        
        # Window and level parameters for display
        self.level = np.mean(self.image)
        self.window = np.max(self.image) - np.min(self.image)
        if self.window < 1:
            self.window = 1
    
    def load_initial_marks(self, init_marks):
        """
        Load initial soma markers from a file or array.
        
        Parameters:
        -----------
        init_marks : str or ndarray
            Path to JSON file with soma data or binary mask array
        """
        if isinstance(init_marks, str) and os.path.exists(init_marks):
            if init_marks.endswith('.json'):
                try:
                    with open(init_marks, 'r') as f:
                        data = json.load(f)
                        self.somas = data.get('somas', [])
                except Exception as e:
                    print(f"Error loading JSON file: {str(e)}")
            else:
                try:
                    # Try to load as a binary image
                    mask = tifffile.imread(init_marks)
                    self._extract_somas_from_mask(mask)
                except Exception as e:
                    print(f"Error loading mask file: {str(e)}")
        elif isinstance(init_marks, np.ndarray):
            self._extract_somas_from_mask(init_marks)
    
    def _extract_somas_from_mask(self, mask):
        """
        Extract somas from a binary mask.
        
        Parameters:
        -----------
        mask : ndarray
            Binary mask where non-zero values indicate somas
        """
        # Convert binary mask to soma list
        labeled, num_features = ndimage.label(mask)
        for i in range(1, num_features + 1):
            coords = np.where(labeled == i)
            if len(coords[0]) > 0:
                center_z, center_y, center_x = [np.mean(coord) for coord in coords]
                # Estimate radius as the maximum distance from center to any point
                distances = np.sqrt(
                    (coords[0] - center_z)**2 + 
                    (coords[1] - center_y)**2 + 
                    (coords[2] - center_x)**2
                )
                radius = np.max(distances)
                self.somas.append([int(center_x), int(center_y), int(center_z), int(radius), self.z_factor])
    
    def generate_senpai_compatible_mask(self):
        """
        Generate a binary mask similar to senpai_somamark output.
        Marked regions are 1, background is 0.
        This mask has the same dimensions as the input image's first three dimensions.
        """
        # Ensure image_shape is based on the first 3 dimensions (Z, Y, X)
        mask_shape = self.image.shape[:3]
        mask = np.zeros(mask_shape, dtype=np.uint8)

        if not self.somas:
            return mask

        # Create grid indices once
        # Ensure these match the order Z, Y, X consistently with image.shape[:3]
        z_indices, y_indices, x_indices = np.ogrid[
            :mask_shape[0],
            :mask_shape[1],
            :mask_shape[2]
        ]

        for soma_params in self.somas:
            # Ensure we unpack correctly if the structure of somas list items changes.
            # Assuming somas store [x, y, z, r, zf]
            x, y, z, r, zf = soma_params
            
            # Ensure coordinates and radius are integers
            x, y, z, r = int(x), int(y), int(z), int(r)
            zf = int(zf) # z_factor should also be int

            # Condition for points within the ellipsoid
            # This formula matches the MATLAB script's logic:
            # (xg - xc)^2 + (yg - yc)^2 + ((zg - zc) * z_fact)^2 <= radius_in_xy^2
            condition = (
                (x_indices - x)**2 +
                (y_indices - y)**2 +
                ((z_indices - z) * zf)**2
            ) <= r**2
            
            mask[condition] = 1  # Set voxels belonging to this soma to 1
        
        return mask

    def on_save(self, event):
        """Callback for the save button.
        Saves soma list to JSON and the Senpai-compatible mask to TIFF.
        """
        # Save the soma markers to a JSON file
        json_path = self.save_somas(self.output_path) # self.save_somas already handles path creation

        if json_path: # Proceed only if JSON saving was successful
            # Generate the Senpai-compatible binary mask
            senpai_mask = self.generate_senpai_compatible_mask()
            
            # Define the output path for the mask TIFF file
            # Example: if json_path is 'image_somas.json', mask_path will be 'image_somas_mask.tif'
            base_path, _ = os.path.splitext(json_path)
            mask_output_path = base_path + '_mask.tif'
            
            try:
                tifffile.imwrite(mask_output_path, senpai_mask, imagej=True) # imagej=True can be useful for metadata
                print(f"Saved Senpai-compatible soma mask to {mask_output_path}")
            except Exception as e:
                print(f"Error saving Senpai-compatible soma mask: {str(e)}")

    def start_gui(self):
        """
        Create and display the interactive GUI for marking somas.
        This method blocks until the GUI window is closed.
        
        Returns:
        --------
        soma_mask : ndarray
            A 3D binary mask of the marked somas (Senpai-compatible).
        """
        # ... (rest of the GUI setup code: self.fig, self.ax, sliders, buttons, etc.)
        # Create the main figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # Display the initial slice
        self.update_display()
        
        # Create slider for Browse slices
        self.ax_slice = plt.axes([0.1, 0.1, 0.8, 0.03])
        self.slice_slider = Slider(
            self.ax_slice, 'Slice', 0, self.image.shape[0]-1,
            valinit=self.slice_idx, valstep=1
        )
        self.slice_slider.on_changed(self.on_slice_change)
        
        # Create button for marking somas
        self.ax_mark = plt.axes([0.1, 0.01, 0.2, 0.05])
        self.mark_button = Button(self.ax_mark, 'Mark Soma')
        self.mark_button.on_clicked(self.on_mark)
        
        # Create button for undoing last soma
        self.ax_undo = plt.axes([0.35, 0.01, 0.1, 0.05])
        self.undo_button = Button(self.ax_undo, 'Undo')
        self.undo_button.on_clicked(self.on_undo)
        
        # Create text boxes for radius and z-factor
        self.ax_radius = plt.axes([0.5, 0.01, 0.1, 0.05])
        self.radius_textbox = TextBox(self.ax_radius, 'Radius:', initial=str(self.radius))
        self.radius_textbox.on_submit(self.on_radius_change)
        
        self.ax_zfactor = plt.axes([0.7, 0.01, 0.1, 0.05])
        self.zfactor_textbox = TextBox(self.ax_zfactor, 'Z-factor:', initial=str(self.z_factor))
        self.zfactor_textbox.on_submit(self.on_zfactor_change)
        
        # Create button for saving somas
        self.ax_save = plt.axes([0.85, 0.01, 0.1, 0.05])
        self.save_button = Button(self.ax_save, 'Save')
        self.save_button.on_clicked(self.on_save) # This will now save JSON and mask
        
        # Connect to mouse events
        self.coords = None
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.scroll_cid = self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        plt.show() # This blocks until the window is closed
        
        # After the GUI is closed, generate and return the Senpai-compatible mask
        soma_mask = self.generate_senpai_compatible_mask()
        return soma_mask
    
    def update_display(self):
        """Update the display with the current slice and markers."""
        if self.ax is None:
            return
            
        self.ax.clear()
        
        # Display the current slice
        if self.image.shape[3] == 1:  # Grayscale
            self.ax.imshow(
                self.image[self.slice_idx, :, :, 0],
                cmap='gray',
                vmin=self.level - self.window/2,
                vmax=self.level + self.window/2
            )
        else:  # RGB
            # Handle potential indices out of bounds
            valid_slice = min(self.slice_idx, self.image.shape[0] - 1)
            self.ax.imshow(self.image[valid_slice])
        
        # Display the current slice number
        self.ax.set_title(f'Slice {self.slice_idx + 1} / {self.image.shape[0]}')
        
        # Mark existing somas in this slice
        for i, (x, y, z, r, zf) in enumerate(self.somas):
            # Calculate the radius at this z-slice (accounting for z-factor)
            z_diff = (z - self.slice_idx) * zf
            slice_radius_sq = r**2 - z_diff**2
            
            if slice_radius_sq > 0:
                slice_radius = np.sqrt(slice_radius_sq)
                circle = plt.Circle((x, y), slice_radius, color='r', fill=False)
                self.ax.add_patch(circle)
                # Add ID number
                self.ax.text(x, y, str(i+1), color='white', 
                           fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle="circle", fc="red", alpha=0.6))
        
        if self.fig is not None:
            self.fig.canvas.draw_idle()
    
    def on_slice_change(self, val):
        """Callback for the slice slider."""
        self.slice_idx = int(val)
        self.update_display()
    
    def on_click(self, event):
        """Callback for mouse clicks."""
        if event.inaxes != self.ax:
            return
        self.coords = (event.xdata, event.ydata)
    
    def on_mark(self, event):
        """Callback for the mark button."""
        if self.coords is None:
            return
        
        x, y = self.coords
        z = self.slice_idx
        r = self.radius
        zf = self.z_factor
        
        self.somas.append([int(x), int(y), int(z), int(r), int(zf)])
        print(f"Marked soma {len(self.somas)} at ({x:.1f}, {y:.1f}, {z})")
        
        self.update_display()
    
    def on_undo(self, event):
        """Callback for the undo button."""
        if self.somas:
            removed = self.somas.pop()
            print(f"Removed soma at ({removed[0]:.1f}, {removed[1]:.1f}, {removed[2]})")
            self.update_display()

    def on_scroll(self, event):
        """Callback for mouse scroll to change slice."""
        if event.button == 'up':
            new_idx = min(self.slice_idx + 1, self.image.shape[0] - 1)
        elif event.button == 'down':
            new_idx = max(self.slice_idx - 1, 0)
        else:
            return
        if new_idx != self.slice_idx:
            self.slice_idx = new_idx
            if self.slice_slider:
                self.slice_slider.set_val(self.slice_idx)
            else:
                self.update_display()
                
    def on_radius_change(self, val):
        """Callback for the radius textbox."""
        try:
            self.radius = int(val)
            if self.radius < 1:
                self.radius = 1
                self.radius_textbox.set_val(str(self.radius))
        except ValueError:
            self.radius_textbox.set_val(str(self.radius))
    
    def on_zfactor_change(self, val):
        """Callback for the z-factor textbox."""
        try:
            self.z_factor = int(val)
            if self.z_factor < 1:
                self.z_factor = 1
                self.zfactor_textbox.set_val(str(self.z_factor))
        except ValueError:
            self.zfactor_textbox.set_val(str(self.z_factor))
    
    def save_somas(self, output_path=None):
        """
        Save the soma markers to a JSON file.
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the output file (defaults to self.output_path)
            
        Returns:
        --------
        str or None
            The path to the saved JSON file, or None if saving failed.
        """
        path = output_path or self.output_path
        
        # Ensure the directory exists
        # os.path.abspath ensures that dirname works correctly even if path is just a filename
        output_dir = os.path.dirname(os.path.abspath(path))
        if not output_dir: # If path was just a filename, dirname might be empty
            output_dir = '.' 
        os.makedirs(output_dir, exist_ok=True)
        
        output_data = {
            'image_path': self.image_path,
            'somas': self.somas,
            'num_somas': len(self.somas),
            'image_dimensions': self.image.shape[:3] # Store Z,Y,X dimensions
        }
        
        try:
            with open(path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Saved {len(self.somas)} soma markers to {path}")
            return path
        except Exception as e:
            print(f"Error saving soma markers JSON: {str(e)}")
            return None
    
    def add_soma(self, x, y, z, radius=None, z_factor=None):
        """
        Programmatically add a soma marker.
        
        Parameters:
        -----------
        x, y, z : int
            Coordinates of the soma center
        radius : int, optional
            Radius of the soma (defaults to self.radius)
        z_factor : int, optional
            Z-axis scaling factor (defaults to self.z_factor)
        """
        r = radius or self.radius
        zf = z_factor or self.z_factor
        self.somas.append([int(x), int(y), int(z), int(r), int(zf)])
        self.update_display()
        return len(self.somas) - 1  # Return the index of the added soma
    
    def remove_soma(self, index=-1):
        """
        Remove a soma marker.
        
        Parameters:
        -----------
        index : int
            Index of the soma to remove (defaults to the last one)
        """
        if 0 <= index < len(self.somas):
            removed = self.somas.pop(index)
            self.update_display()
            return removed
        return None
    
    def generate_binary_mask(self):
        """
        Generate a binary mask from the soma markers.
        
        Returns:
        --------
        mask : ndarray
            3D binary mask where each soma has a unique ID
        """
        mask = np.zeros(self.image.shape[:3], dtype=np.uint8)
        
        z_indices, y_indices, x_indices = np.ogrid[
            :self.image.shape[0], 
            :self.image.shape[1], 
            :self.image.shape[2]
        ]
        
        for i, (x, y, z, r, zf) in enumerate(self.somas):
            # Create a spherical mask
            sphere_mask = (
                (x_indices - x)**2 + 
                (y_indices - y)**2 + 
                ((z_indices - z) * zf)**2
            ) <= r**2
            
            # Set the mask to the soma ID
            mask[sphere_mask] = i + 1
        
        return mask
    
    def load_from_json(self, json_path):
        """
        Load soma markers from a JSON file.
        
        Parameters:
        -----------
        json_path : str
            Path to the JSON file
        """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.somas = data.get('somas', [])
            self.update_display()
            return self.somas
        except Exception as e:
            print(f"Error loading soma markers from {json_path}: {str(e)}")
            return []
    
    def get_somas_as_dataframe(self):
        """
        Get the soma markers as a pandas DataFrame.
        
        Returns:
        --------
        df : pandas.DataFrame
            DataFrame with columns ['x', 'y', 'z', 'radius', 'z_factor', 'id']
        """
        try:
            data = []
            for i, (x, y, z, r, zf) in enumerate(self.somas):
                data.append({
                    'id': i + 1,
                    'x': x,
                    'y': y,
                    'z': z,
                    'radius': r,
                    'z_factor': zf
                })
            return pd.DataFrame(data)
        except ImportError:
            print("pandas is not installed, returning list instead")
            return self.somas
    
    def get_soma_centers(self):
        """
        Get the centers of all somas.
        
        Returns:
        --------
        centers : ndarray
            Array of shape (n_somas, 3) with the (x, y, z) coordinates of each soma
        """
        if not self.somas:
            return np.array([])
        
        # Extract just the x, y, z coordinates from each soma
        centers = np.array([[x, y, z] for x, y, z, _, _ in self.somas])
        return centers


def mark_somas_gui(image_path=None, image_data=None, **kwargs):
    """
    Convenience function to start the GUI for marking somas.
    
    Parameters:
    -----------
    image_path : str, optional
        Path to the 3D TIFF image stack
    image_data : ndarray, optional
        3D or 4D image array, if provided directly instead of path
    **kwargs : 
        Additional parameters to pass to NeuronalSomaMarker
        
    Returns:
    --------
    soma_mask : ndarray
        A 3D binary mask of the marked somas (Senpai-compatible).
    """
    marker = NeuronalSomaMarker(image_data=image_data, image_path=image_path, **kwargs)
    return marker.start_gui() # This now returns the Senpai-compatible mask


def batch_process_images(image_dir, output_dir=None, pattern='*.tif', **kwargs):
    """
    Process multiple images in batch mode. For each image, a GUI will open.
    Saves JSON and Senpai-compatible mask TIFF via the GUI's "Save" button.
    
    Parameters:
    -----------
    image_dir : str
        Directory containing image files
    output_dir : str, optional
        Directory to save output files (defaults to image_dir)
    pattern : str, optional
        Glob pattern for finding image files (default: '*.tif')
    **kwargs :
        Additional parameters to pass to NeuronalSomaMarker
    
    Returns:
    --------
    results : dict
        Dictionary mapping image paths to *base* output paths (e.g., '/path/to/image_somas').
        The actual files will be like 'image_somas.json' and 'image_somas_mask.tif'.
    """
    from glob import glob # Keep this import local if not used elsewhere at module level
    
    if output_dir is None:
        output_dir = image_dir
    os.makedirs(output_dir, exist_ok=True)
    
    image_paths = glob(os.path.join(image_dir, pattern))
    results = {}
    
    for image_path in image_paths:
        try:
            basename = os.path.basename(image_path)
            # Define the base for output filenames (without extension)
            output_base_name = os.path.splitext(basename)[0] + '_somas'
            output_json_path = os.path.join(output_dir, output_base_name + '.json')
            
            print(f"Processing {basename}...")
            # Pass the intended JSON path to NeuronalSomaMarker. The on_save method
            # will use this to derive the mask filename.
            marker = NeuronalSomaMarker(
                image_path=image_path, 
                output_path=output_json_path, # This is now the JSON path
                **kwargs
            )
            # start_gui will open the interactive window.
            # The user needs to click "Save" in the GUI to generate output files.
            # The returned mask from start_gui is not explicitly used here,
            # as saving is handled by the GUI's save button.
            marker.start_gui() 
            
            # The "Save" button saves both JSON and mask.
            # We store the base path for the results dictionary.
            results[image_path] = os.path.join(output_dir, output_base_name)
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    return results

# Make sure the main() function reflects these changes if needed, particularly
# how it handles the output of mark_somas_gui or batch_process_images.
# For example, if you run a single image, you might want to save the returned mask.

def main():
    """Command-line interface for the soma marker."""
    parser = argparse.ArgumentParser(description='Mark neuronal somas in 3D image stacks.')
    # ... (argument parsing remains the same) ...
    parser.add_argument('image_path', type=str, help='Path to the 3D TIFF image stack or directory for batch mode')
    parser.add_argument('--output', '-o', type=str, help='Path to save the output JSON file (and corresponding _mask.tif). For batch mode, this is the output directory.')
    parser.add_argument('--init-marks', '-i', type=str, help='Path to initial soma markers (JSON or mask TIFF)')
    parser.add_argument('--radius', '-r', type=int, default=10, help='Radius of the sphere (default: 10)')
    parser.add_argument('--z-factor', '-z', type=int, default=3, help='Z-axis scaling factor (default: 3)')
    parser.add_argument('--batch', '-b', action='store_true', help='Process all images in the directory specified by image_path')
    parser.add_argument('--pattern', '-p', type=str, default='*.tif', help='Glob pattern for batch processing (e.g., "*.tif", "*.tiff") (default: *.tif)')
    
    args = parser.parse_args()
    
    if args.batch:
        image_dir = args.image_path # In batch mode, image_path is the directory
        output_dir = args.output or image_dir # If output not specified, use image_dir
        print(f"Batch processing images in directory: {image_dir}")
        print(f"Output will be saved to directory: {output_dir}")
        batch_process_images(
            image_dir=image_dir, 
            output_dir=output_dir,
            pattern=args.pattern,
            init_marks=args.init_marks,
            radius=args.radius,
            z_factor=args.z_factor
        )
    else:
        # For single image processing
        # The output_path for the marker constructor is the JSON path.
        # The mask will be saved with _mask.tif suffix relative to this.
        output_json_path = args.output
        if not output_json_path: # If no output path specified, create one based on image_path
             output_json_path = os.path.splitext(args.image_path)[0] + '_somas.json'

        print(f"Processing single image: {args.image_path}")
        print(f"Output JSON will be: {output_json_path}")
        print(f"Output mask will be: {os.path.splitext(output_json_path)[0] + '_mask.tif'}")

        soma_marker = NeuronalSomaMarker(
            image_path=args.image_path,
            output_path=output_json_path, # This is the path for the JSON file
            init_marks=args.init_marks,
            radius=args.radius,
            z_factor=args.z_factor
        )
        returned_mask = soma_marker.start_gui()
        
        # If the user didn't click "Save" in the GUI, the returned_mask contains the current state.
        # You might want to offer to save it here if no output file was created by the "Save" button.
        # For now, we assume the user will use the "Save" button.
        if returned_mask is not None and returned_mask.any():
             print("GUI closed. If you used the 'Save' button, files were saved.")
             # Example: save if not saved via button (though button is preferred)
             # mask_output_path_check = os.path.splitext(output_json_path)[0] + '_mask.tif'
             # if not os.path.exists(mask_output_path_check):
             #    tifffile.imwrite(mask_output_path_check, returned_mask, imagej=True)
             #    print(f"Returned mask saved to {mask_output_path_check} as GUI was closed without explicit save.")

if __name__ == '__main__':
    main()