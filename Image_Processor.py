import nd2
import numpy as np
from skimage.io import imread
from skimage import filters
import napari
import tifffile as tiff
import os
import argparse
from magicgui import magicgui


class ImageProcessor:
    def __init__(self, image_path = None, file_save='somas_data.npy', data_path='data/', channel=None):
        
        self.data_path = data_path
        self.data_folder()
        self.channel = channel
        self.file_save = file_save
        self.image_path = image_path
        self.read_image(image_path)

        # Napari viewer defaults
        self.processed_image = None
        self.viewer = None
        self.points_layer = None

        # Default parameters of magicgui widgets
        self.xy_radius = 10
        self.z_radius = 3


    def data_folder(self):
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
    
    def read_image(self, image_path):
        if image_path.endswith('.tif'):
            self.tiff_preprocess()
        else:
            if image_path.endswith('.nd2'):
                self.nd2_preprocess()
            else:
                raise ValueError("Unsupported file format. Please use .tif or .nd2 files.")
            
    def tiff_preprocess(self):
        with tiff.TiffFile(os.path.join(self.data_path, self.image_path)) as tif:
            page = tif.pages[0]
            tags = page.tags

            Img_width = tags.get("ImageWidth")
            Img_length = tags.get("ImageLength")
            Img_Description = tags.get("ImageDescription")
            XYRes = tags.get("XResolution")

            Img_Description.value.split("\n")
            slices = int([line.split("=")[1] for line in Img_Description.value.split("\n") if line.startswith("images=")][0])
            spacing = float([line.split("=")[1] for line in Img_Description.value.split("\n") if line.startswith("spacing=")][0])
            
            self.xy_res = XYRes.value[1] / XYRes.value[0]  # µm/pixel
            self.z_res = spacing # µm/pixel in Z
            self.width_px = Img_width.value
            self.height_px = Img_length.value
            self.z_px = slices
            self.image_data = tif.asarray()

    def nd2_preprocess(self):
        with nd2.ND2File(os.path.join(self.data_path, self.image_path)) as f:
            channel_info = [
                {
                    "name": ch.channel.name,
                    "emission_nm": ch.channel.emissionLambdaNm,
                    "color": (ch.channel.color.r, ch.channel.color.g, ch.channel.color.b)
                }
                for ch in f.metadata.channels
            ]
            for i in range(len(channel_info)):
                print(f"{channel_info[i]}\n")
            # Pixel dimensions (Y, X)
            shape = f.sizes 
            width_px = shape.get('X')
            height_px = shape.get('Y')
            z_px = shape.get('Z')

            # Physical pixel size in micrometers
            print(f.voxel_size())  # VoxelSize(X=0.325, Y=0.325, Z=1.0)
            px_size_xy = f.voxel_size().x  # µm/pixel in X
            px_size_z = f.voxel_size().z   # µm/pixel in Z

        print(f"Image size: ({width_px} x {height_px} px) x {z_px}")
        print(f"Pixel size: {px_size_xy:.4f} µm (XY), {px_size_z:.4f} µm (Z)")

        image_data = nd2.imread(os.path.join(self.data_path, self.image_path))
        if self.channel is not None:
            image_data = image_data[:, self.channel, :, :]
        else:
            raise ValueError("Channel index is required for ND2 files.")

        self.image_data = image_data
        self.width_px = width_px
        self.height_px = height_px
        self.z_px = z_px
        self.xy_res = px_size_xy
        self.z_res = px_size_z

    def _on_point_added(self, event):
        """Event handler that creates an ellipsoid when a point is added."""
        points = event.source.data
        if len(points) == 0:
            return  # No points to process
        # Add an ellipsoid around the last added point
        center_point = points[-1]
        for z in range(-self.z_radius, self.z_radius+1):
            x = int(self.xy_radius * np.sqrt(1 - (z / self.z_radius) ** 2))
            bounding_box = np.array([
                [center_point[0] + z, center_point[1] + x, center_point[2] - x],
                [center_point[0] + z, center_point[1] + x, center_point[2] + x],
                [center_point[0] + z, center_point[1] - x, center_point[2] + x],
                [center_point[0] + z, center_point[1] - x, center_point[2] - x],
            ])
            self.shapes_layer.add(bounding_box, shape_type='ellipse')

    @magicgui(
            layout="vertical",
            # We use sliders for intuitive control over the radii
            xy_radius={"widget_type": "Slider", "min": 1, "max": 50},
            z_radius={"widget_type": "Slider", "min": 1, "max": 10},
            auto_call=True,
        )

    def _slider_widget(self, xy_radius: int = 15, z_radius: int = 5):
        """Creates the GUI widget that controls this class instance."""
        # Update the instance's radius attributes whenever the slider changes
        self.xy_radius = xy_radius
        self.z_radius = z_radius

        return self._slider_widget
    
    @magicgui(
            layout="vertical",
            # This button is used to trigger the saving method
            save_button ={"widget_type": "PushButton", "text": "Save Somas"},
            auto_call=False,
            call_button=False,
        )
    
    def _save_widget(self, save_button):
        """Creates the GUI widget that controls this class instance."""
        self._save_widget.save_button.clicked.connect(self.save_somas_data)
        return self._save_widget

    def open_viewer(self):
        self.viewer = napari.Viewer()
        self.viewer.add_image(self.image_data)
        self.shapes_layer = self.viewer.add_shapes(name='Soma Ellipsoids', edge_color='magenta', face_color='transparent', ndim=3,)
        self.points_layer = self.viewer.add_points(name='Soma Centroids', face_color='cyan', size=5, ndim=3)
        self.points_layer.events.data.connect(self._on_point_added)
        self.viewer.window._qt_window.closeEvent = self._on_close
        #napari.view_image()
        self.viewer.window.add_dock_widget(self._slider_widget(), name="Soma Controls", area='right')
        self.viewer.window.add_dock_widget(self._save_widget(), name="Save Somas", area='right')
        print("Napari window opened. Add somas (points) and close the window when done.")
        napari.run()

    def _on_close(self, event):
        """Triggered when the user closes the window."""
        print("Viewer closed. Saving metadata...")
        #self.save_data()
        self.save_metadata()
        event.accept()


    def save_somas_data(self):
        """Save soma coordinates to a .npy file."""
        if self.points_layer is not None:
            np.save(self.file_save, self.points_layer.data)
            print(f"Saved {len(self.points_layer.data)} soma points to {self.file_save}")
        else:
            print("No points layer found — nothing to save.")
        
        if self.shapes_layer is not None:
            ellipsoids = []
            for shape in self.shapes_layer.data:
                min_corner = shape.min(axis=0)
                max_corner = shape.max(axis=0)
                center = (min_corner + max_corner) / 2
                radii = (max_corner - min_corner) / 2
                ellipsoids.append((center, radii))
            np.save(self.file_save.replace('.npy', 'soma_ellipsoids.npy'), ellipsoids)
            print(f"Saved {len(ellipsoids)} soma ellipsoids to {self.file_save.replace('.npy', 'soma_ellipsoids.npy')}")

    def get_points(self):
        """Return the soma coordinates after annotation."""
        if self.points_layer is not None:
            return np.array(self.points_layer.data)
        print("No points layer found - returning empty array.")
        return np.empty((0, self.image_data.ndim))
    
    def save_metadata(self):
        """Save the processed metadata and image data."""
        metadata = {
            'image_path': self.image_path,
            'file_save': self.file_save,
            'data_path': self.data_path,
            'channel': self.channel,
            'width_px': self.width_px,
            'height_px': self.height_px,
            'z_px': self.z_px,
            'xy_res': self.xy_res,
            'z_res': self.z_res
        }
        np.save(os.path.join(self.data_path, 'processed_metadata.npy'), metadata)
        np.save(os.path.join(self.data_path, 'processed_image.npy'), self.image_data)
        print(f"Processed metadata and image data saved in {self.data_path}")


def main():
    """Main function to run the annotation script from the command line."""
    parser = argparse.ArgumentParser(
        description="Annotate 3D microscopy images with points using napari."
    )
    
    # Required argument
    parser.add_argument(
        "-i", "--image_path",
        type=str,
        help="Path to the .nd2 or .tif image file."
    )
    
    # Optional arguments
    parser.add_argument(
        "-c", "--channel",
        type=int,
        default=0,
        help="Channel index to load for multi-channel images (default: 0)."
    )
    parser.add_argument(
        "-d", "--data-path",
        type=str,
        default="data/",
        help="Directory to save the output .npy file (default: data/)."
    )
    parser.add_argument(
        "-s", "--file-save",
        type=str,
        default="somas_coords.npy",
        help="Filename for the already saved coordinates (default: somas_coords.npy)."
    )
    """parser.add_argument(
        "-e", "--ellipsoids",
        type=str,
        default="somas_ellipsoids.npy",
        help="Filename for the already saved ellipsoids (default: somas_ellipsoids.npy)."
    )"""
    
    args = parser.parse_args()
    
    try:
        # Initialize the processor with arguments from the terminal
        processor = ImageProcessor(
            image_path=args.image_path,
            file_save=args.file_save,
            data_path=args.data_path,
            channel=args.channel
        )
        # Start the annotation process
        processor.open_viewer()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()