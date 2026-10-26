import nd2
import numpy as np
from skimage.io import imread
from skimage import filters
import napari
import tifffile as tiff
import os
import argparse
from magicgui import magicgui
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.transform import downscale_local_mean


def detect_somas(image, xy_res, z_res, radius_um=8.0, voxel_um=1.0, background_um=20.0):
    """Automatic soma proposals: Gaussian smoothing (1 um) minus a large-scale background (Gaussian, background_um;
    None to skip) -> Otsu threshold -> opening with a ball of 0.6*radius_um -> local maxima of the distance transform.
    The stack is block-averaged to ~voxel_um voxels first. The background subtraction removes uneven illumination,
    which otherwise makes Otsu select whole bright regions (test_asia: 46% of the volume).
    image is (Z, Y, X); returns an (N, 4) array [z, y, x, r]: centre in pixels of `image` and the distance-transform
    value at the peak in um (~ soma radius), sorted by r (largest first)."""
    fz = max(1, int(round(voxel_um / z_res)))
    fxy = max(1, int(round(voxel_um / xy_res)))
    small = downscale_local_mean(np.asarray(image, dtype=np.float32), (fz, fxy, fxy))
    res = np.array([z_res * fz, xy_res * fxy, xy_res * fxy])
    smooth = ndimage.gaussian_filter(small, 1.0 / res)
    if background_um:
        smooth -= ndimage.gaussian_filter(small, background_um / res)
    fg = smooth > filters.threshold_otsu(smooth)
    # Opening with a Euclidean ball (erosion then dilation), done with distance transforms: much faster than binary_opening
    r_open = 0.6 * radius_um
    core = ndimage.distance_transform_edt(fg, sampling=res) > r_open
    if not core.any():
        return np.empty((0, 4))
    opened = ndimage.distance_transform_edt(~core, sampling=res) <= r_open
    edt = ndimage.distance_transform_edt(opened, sampling=res)
    peaks = peak_local_max(edt, min_distance=max(2, int(round(1.2 * radius_um / res[1]))),
                           threshold_abs=0.5, exclude_border=False)
    r = edt[tuple(peaks.T)]
    order = np.argsort(-r)
    centres = (peaks[order] + 0.5) * np.array([fz, fxy, fxy]) - 0.5
    return np.column_stack([centres, r[order]])


class ImageProcessor:
    def __init__(self, image_path = None, data_path='data/', channel=None):
        
        self.data_path = data_path
        self.channel = channel
        self.file_save = 'somas_data.npy'
        self.image_path = image_path
        self.folder_save = os.path.join(data_path, image_path.split('.')[0])
        self.check = self.check_already_processed()
        self.read_image(image_path)
        self.data_folder()
        self.auto_somas = self.load_or_detect_somas()

        # Napari viewer defaults
        self.processed_image = None
        self.viewer = None
        self.points_layer = None

        # Default parameters of magicgui widgets
        self.xy_radius = 10
        self.z_radius = 3

    def data_folder(self):
        if not os.path.exists(self.folder_save):
            os.makedirs(self.folder_save)

    def load_or_detect_somas(self):
        """Load the automatic soma proposals, or run detect_somas once and save them to auto_somas.npy ([z, y, x, r_um])."""
        auto_file = os.path.join(self.folder_save, 'auto_somas.npy')
        if os.path.exists(auto_file):
            somas = np.load(auto_file)
            print(f"Loaded {len(somas)} automatic soma proposals from {auto_file}")
            return somas
        print("Running automatic soma detection (background-subtracted Otsu + opening + distance-transform peaks, r=8 um)...")
        somas = detect_somas(self.image_data, self.xy_res, self.z_res)
        np.save(auto_file, somas)
        print(f"Saved {len(somas)} automatic soma proposals to {auto_file}")
        return somas

    def read_image(self, image_path):
        if self.check == True:
            return
        if image_path.endswith('.tif'):
            self.tiff_preprocess()
        else:
            if image_path.endswith('.nd2'):
                self.nd2_preprocess()
            else:
                raise ValueError("Unsupported file format. Please use .tif or .nd2 files.")

    def check_already_processed(self):
        metadata_file = os.path.join(self.folder_save, 'processed_metadata.npy')
        image_file = os.path.join(self.folder_save, 'processed_image.npy')
        if os.path.exists(metadata_file) and os.path.exists(image_file):
            metadata = np.load(metadata_file, allow_pickle=True).item()
            self.image_data = np.load(image_file)
            self.width_px = metadata['width_px']
            self.height_px = metadata['height_px']
            self.z_px = metadata['z_px']
            self.xy_res = metadata['xy_res']
            self.z_res = metadata['z_res']
            print("Loaded previously processed image and metadata.")
            if os.path.exists(os.path.join(self.folder_save, self.file_save)):
                print("Found previously saved soma coordinates.")
                self.points_layer_data = np.load(os.path.join(self.folder_save, self.file_save))
                self.shapes_layer_data = np.load(os.path.join(self.folder_save, 'soma_napari_ellipsoids.npy'))
                return True
        return False
    

    def tiff_preprocess(self):
        self.img_path_full = os.path.join(self.data_path, 'imgs', self.image_path)
        with tiff.TiffFile(self.img_path_full) as tif:
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
        self.img_path_full = os.path.join(self.data_path, 'imgs', self.image_path)
        with nd2.ND2File(self.img_path_full) as f:
            channel_info = [
                {
                    "name": ch.channel.name,
                    "emission_nm": ch.channel.emissionLambdaNm,
                    "color": (ch.channel.color.r, ch.channel.color.g, ch.channel.color.b)
                }
                for ch in f.metadata.channels
            ]
            for i in range(len(channel_info)):
                name = channel_info[i]['name']
                emission_nm = channel_info[i]['emission_nm']
                color = channel_info[i]['color']
                print(f"Channel {i}: {name}, Emission: {emission_nm} nm, Color: {color}")
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

        image_data = nd2.imread(self.img_path_full)
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

    #def create_bounding_box(self, create_)

    def _on_point_added(self, event,last_point = False):
        """Event handler that creates an ellipsoid when a point is added.""" 
        print(event.action)
        if event.action != 'added':
            return
        
        points = self.points_layer.data
        if len(points) == 0:
            return  # No points to process
        
        if last_point is True:
            length_shapes = len(self.shapes_layer.data)
            self.shapes_layer.remove(list(range(length_shapes - len(self.z_radius_list), length_shapes)))  # Clear last shape
            
        # Add an ellipsoid around the last added point
        center_point = points[-1]
        z_radius = np.arange(-self.z_radius, self.z_radius+1)
        x_radius = [int(self.xy_radius * np.sqrt(1 - (z / self.z_radius) ** 2)) for z in z_radius]
        self.x_radius_list = x_radius
        self.z_radius_list = z_radius
        for x, z in zip(x_radius, z_radius):
            bounding_box = np.array([
                [center_point[0] + z, center_point[1] + x, center_point[2] - x],
                [center_point[0] + z, center_point[1] + x, center_point[2] + x],
                [center_point[0] + z, center_point[1] - x, center_point[2] + x],
                [center_point[0] + z, center_point[1] - x, center_point[2] - x],
            ])
            self.shapes_layer.add(bounding_box, shape_type='ellipse')

        print(len(self.shapes_layer.data))

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
        

    
    @magicgui(
            layout="vertical",
            # This button is used to trigger the saving method
            save_button ={"widget_type": "PushButton", "text": "Save Somas"},
            auto_call=True,
        )
    
    def _save_widget(self, save_button):
        """Creates the GUI widget that controls this class instance."""
        #return self._save_widget
    
    @magicgui(
            layout="vertical",
            # This button is used to trigger the loading method
            load_button ={"widget_type": "PushButton", "text": "Load Somas"},
        )
    def _load_widget(self, load_button):
        """Creates the GUI widget that controls this class instance."""
    
    @magicgui(
            layout="vertical",
            # This button is used to trigger the undo method
            undo_button ={"widget_type": "PushButton", "text": "Undo Last Soma"},
            auto_call=True,
        )
    
    def _undo_widget(self, undo_button):
        """Creates the GUI widget that controls this class instance."""

    def open_viewer(self):
        self.viewer = napari.Viewer()
        self.viewer.add_image(self.image_data)
        if len(self.auto_somas):
            # Proposals only (not saved as somas); added before 'Soma Centroids' so clicks still go to the manual layer
            self.viewer.add_points(self.auto_somas[:, :3], name='Auto Somas (proposed)', ndim=3,
                                   size=2 * self.auto_somas[:, 3] / self.xy_res, face_color='transparent',
                                   border_color='yellow', opacity=0.8)

        self.points_layer = self.viewer.add_points(name='Soma Centroids', face_color='magenta', size=5, ndim=3)
        self.points_layer.events.data.connect(self._on_point_added)
        self.shapes_layer = self.viewer.add_shapes(name='Soma Ellipsoids', edge_color='red', face_color='red', ndim=3, opacity=0.5)
        self.viewer.window._qt_window.closeEvent = self._on_close
        #napari.view_image()
        self.viewer.window.add_dock_widget(self._slider_widget, name="Soma Controls", area='right')
        self.viewer.window.add_dock_widget(self._undo_widget, name="Undo Soma", area='right')
        self.viewer.window.add_dock_widget(self._save_widget, name="Save Somas", area='right')
        if self.check == True:
            self.viewer.window.add_dock_widget(self._load_widget, name="Load Somas", area='right')
        self._load_widget.load_button.clicked.connect(self.load_somas_data)
        self._save_widget.save_button.clicked.connect(self.save_somas_data)
        self._undo_widget.undo_button.clicked.connect(self.undo_soma)
        #self._slider_widget.update_soma_radii.clicked.connect(self._on_point_added(last_point=True))

        print("Napari window opened. Add somas (points) and close the window when done.")
        napari.run()

    def _on_close(self, event):
        """Triggered when the user closes the window."""
        print("Viewer closed. Saving metadata...")
        #self.save_data()
        self.save_metadata()
        event.accept()

    def undo_soma(self):
        if len(self.points_layer.data) > 0:
            #self.points_layer.events.data.disconnect(self._on_point_added)
            self.points_layer.remove(list(range(len(self.points_layer.data) - 1, len(self.points_layer.data))))
            self.shapes_layer.remove(list(range(len(self.shapes_layer.data) - len(self.z_radius_list), len(self.shapes_layer.data))))
            #self.points_layer.events.data.connect(self._on_point_added)
        else:
            print("No soma points to undo.")

    def load_somas_data(self):
        self.points_layer.data = self.points_layer_data
        self.shapes_layer.data = self.shapes_layer_data

    def save_somas_data(self):
        """Save soma coordinates to a .npy file."""
        if self.points_layer is not None:
            np.save(os.path.join(self.folder_save, self.file_save), self.points_layer.data)
            print(f"Saved {len(self.points_layer.data)} soma points to {self.folder_save}/{self.file_save}")
            
        else:
            print("No points layer found — nothing to save.")
        
        if self.shapes_layer is not None:
            np.save(os.path.join(self.folder_save, 'soma_napari_ellipsoids.npy'), self.shapes_layer.data)
            print(f"Saved {len(self.shapes_layer.data)} soma ellipsoids to {self.folder_save}/soma_napari_ellipsoids.npy")
            # Additionally, create and save a binary mask of the somas
            mask_shape = self.image_data.shape
            soma_mask = np.zeros(mask_shape, dtype=bool)
            y_indices, x_indices = np.ogrid[0:mask_shape[1], 0:mask_shape[2]]
            # Each soma is a stack of 2D ellipses (one shape per z-slice), so rasterise each one as a disc in its own slice
            for coords in self.shapes_layer.data:
                z, y, x = np.round(np.mean(coords, axis=0)).astype(int)
                r = (coords[:, 2].max() - coords[:, 2].min()) / 2
                if 0 <= z < mask_shape[0]:
                    soma_mask[z][(x_indices - x)**2 + (y_indices - y)**2 <= r**2] = True

            tiff.imwrite(os.path.join(self.folder_save, 'soma_mask.tiff'), soma_mask)
            print(f"Saved {len(self.shapes_layer.data)} soma masks to {self.folder_save}/soma_mask.tiff")
    
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
        np.save(os.path.join(self.folder_save, 'processed_metadata.npy'), metadata)
        np.save(os.path.join(self.folder_save, 'processed_image.npy'), self.image_data)
        print(f"Processed metadata and image data saved in {self.folder_save}")


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
    
    args = parser.parse_args()
    
    try:
        # Initialize the processor with arguments from the terminal
        processor = ImageProcessor(
            image_path=args.image_path,
            data_path=args.data_path,
            channel=args.channel
        )
        # Start the annotation process
        
        processor.open_viewer()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()