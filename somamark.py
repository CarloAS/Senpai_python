import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
import tifffile
import json
import os
from scipy import ndimage
import argparse
from pathlib import Path
import matplotlib.cm as cm


class NeuronalSomaMarker:
    """
    A GUI tool for marking neuronal somas in 3D image stacks.
    This is a Python translation of the MATLAB senpai_somamark function.
    """
    
    def __init__(self, image_path, output_path=None, init_marks=None, radius=10, z_factor=3):
        """
        Initialize the NeuronalSomaMarker.
        
        Parameters:
        -----------
        image_path : str
            Path to the 3D TIFF image stack
        output_path : str, optional
            Path to save the output JSON file
        init_marks : ndarray, optional
            Initial soma markers
        radius : int, optional
            Radius of the sphere used to mark somas
        z_factor : int, optional
            Z-axis scaling factor for the sphere
        """
        self.image_path = image_path
        self.output_path = output_path or os.path.splitext(image_path)[0] + '_somas.json'
        
        # Load the image
        self.image = tifffile.imread(image_path)
        if len(self.image.shape) == 3:  # Add channel dimension if grayscale
            self.image = self.image[..., np.newaxis]
        
        # Initialize parameters
        self.slice_idx = self.image.shape[0] // 2
        self.radius = radius
        self.z_factor = z_factor
        self.somas = []  # Will store [x, y, z, radius, z_factor] for each soma
        
        # If initial marks are provided
        if init_marks is not None:
            self.load_initial_marks(init_marks)
        
        # Window and level parameters
        self.level = np.mean(self.image)
        self.window = np.max(self.image) - np.min(self.image)
        if self.window < 1:
            self.window = 1
        
        # Create visualization
        self.create_gui()
        
    def load_initial_marks(self, init_marks_path):
        """Load initial soma markers from a file."""
        if isinstance(init_marks_path, str) and init_marks_path.endswith('.json'):
            with open(init_marks_path, 'r') as f:
                data = json.load(f)
                self.somas = data['somas']
        elif isinstance(init_marks_path, np.ndarray):
            # Convert binary mask to soma list
            labeled, num_features = ndimage.label(init_marks_path)
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
    
    def create_gui(self):
        """Create the GUI elements."""
        # Create the main figure
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.25)
        
        # Display the initial slice
        self.update_display()
        
        # Create slider for browsing slices
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
        self.save_button.on_clicked(self.on_save)
        
        # Connect to mouse events
        self.coords = None
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.show()
    
    def update_display(self):
        """Update the display with the current slice and markers."""
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
            self.ax.imshow(self.image[self.slice_idx])
        
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
    
    def on_save(self, event):
        """Callback for the save button."""
        # Save the soma markers to a JSON file
        output_data = {
            'image_path': self.image_path,
            'somas': self.somas,
            'num_somas': len(self.somas),
            'image_dimensions': self.image.shape[:3]
        }
        
        with open(self.output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved {len(self.somas)} soma markers to {self.output_path}")
    
    def generate_binary_mask(self):
        """Generate a binary mask from the soma markers."""
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


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Mark neuronal somas in 3D image stacks.')
    parser.add_argument('image_path', type=str, help='Path to the 3D TIFF image stack')
    parser.add_argument('--output', '-o', type=str, help='Path to save the output JSON file')
    parser.add_argument('--init-marks', '-i', type=str, help='Path to initial soma markers')
    parser.add_argument('--radius', '-r', type=int, default=10, help='Radius of the sphere (default: 10)')
    parser.add_argument('--z-factor', '-z', type=int, default=3, help='Z-axis scaling factor (default: 3)')
    
    args = parser.parse_args()
    
    # Create the soma marker
    soma_marker = NeuronalSomaMarker(
        args.image_path,
        args.output,
        args.init_marks,
        args.radius,
        args.z_factor
    )


if __name__ == '__main__':
    main()