import nd2
import numpy as np
from skimage.io import imread
from skimage import filters, measure, segmentation
import napari
import tifffile as tiff
import os
import argparse
from magicgui import magicgui
from magicgui.widgets import Container, PushButton
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.transform import downscale_local_mean
from skimage.morphology import remove_small_holes

# Channel orders of Asia's stains (Fiji channel N = index N-1); extra channels are named ch<N>. The stain is taken from '_ID_' or
# '_Mech_' anywhere in the file name (e.g. IF_D14_ID_40x_... / IF_D14_Mech_40x_...); any other name uses the ID order (CHANNEL_NAMES).
CHANNEL_ORDERS = {'id': ['dapi', 'map2', 'actin', 'ki67'], 'mech': ['dapi', 'map2', 'piezo1', 'tuj1']}
CHANNEL_NAMES = CHANNEL_ORDERS['id']
CHANNEL_COLORMAPS = {'dapi': 'blue', 'map2': 'green', 'actin': 'red', 'ki67': 'gray', 'piezo1': 'red', 'tuj1': 'gray'}
SOMA_CHANNEL = 'map2'  # channel used for soma detection in multi-channel TIFFs, unless -c is given
CELL_CLASSES = ['neuron', 'astrocyte', 'other']  # class codes 0, 1, 2 of detect_cells
# Accepted neuron ellipsoids = this fraction of the nucleus radius: they are watershed seeds (NeuronSeparator labels mask components),
# full-size ellipsoids touch in clusters (IF_D14 test: 51 components for 80 neurons at 1.0, 70 at 0.7, 80 at 0.5)
NEURON_SEED_FRACTION = 0.5


def detect_somas(image, xy_res, z_res, radius_um=8.0, voxel_um=1.0, background_um=20.0):
    """Automatic soma proposals: Gaussian smoothing (1 um) minus a large-scale background (Gaussian, background_um;
    None to skip) -> triangle threshold -> per-slice filling of holes smaller than pi*radius_um^2 -> opening with a ball
    of 0.6*radius_um -> local maxima of the distance transform.
    The stack is block-averaged to ~voxel_um voxels first. The background subtraction removes uneven illumination,
    which otherwise makes the threshold select whole bright regions (test_asia: 46% of the volume with Otsu).
    Triangle instead of Otsu keeps dim somas (MAP2 new_asia_test: Otsu keeps 3% of the volume, 19 vs 52 proposals);
    the hole filling closes nuclear voids of MAP2-stained somas, and is size-limited so neurite loops are not filled.
    image is (Z, Y, X); returns an (N, 4) array [z, y, x, r]: centre in pixels of `image` and the distance-transform
    value at the peak in um (~ soma radius), sorted by r (largest first)."""
    fz = max(1, int(round(voxel_um / z_res)))
    fxy = max(1, int(round(voxel_um / xy_res)))
    small = downscale_local_mean(np.asarray(image, dtype=np.float32), (fz, fxy, fxy))
    res = np.array([z_res * fz, xy_res * fxy, xy_res * fxy])
    smooth = ndimage.gaussian_filter(small, 1.0 / res)
    if background_um:
        smooth -= ndimage.gaussian_filter(small, background_um / res)
    fg = smooth > filters.threshold_triangle(smooth)
    max_hole = int(np.pi * radius_um**2 / (res[1] * res[2]))
    fg = np.stack([remove_small_holes(s, area_threshold=max_hole) for s in fg])
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


def channel_order_from_name(image_path):
    """Channel names from '_ID_' or '_Mech_' in the file name (case-insensitive; CHANNEL_ORDERS); neither present: ID order."""
    name = os.path.basename(image_path).lower()
    stain = next((k for k in CHANNEL_ORDERS if f'_{k}_' in name), 'id')
    return CHANNEL_ORDERS[stain]


def detect_cells(channels, xy_res, z_res, voxel_um=0.5, background_um=20.0, min_radius_um=2.5, shell_um=(0.5, 3.0),
                 map2_min=5.0, actin_min=5.0, ki67_min=10.0):
    """Cell proposals from a multi-channel stack (channels: {name: (Z, Y, X)}, needs 'dapi' and 'map2'; 'actin', 'ki67' optional).
    Nuclei: DAPI block-averaged to ~voxel_um in XY, Gaussian 0.7 um minus Gaussian background_um, Otsu, seeds = maxima of DAPI
    smoothed at 1.5 um (min distance 2.5 um), watershed; nuclei with equivalent radius < min_radius_um are dropped (debris).
    Each channel is expressed as a robust z-score ((value - median) / (1.4826 MAD) of its background-subtracted image).
    Per nucleus: MAP2 and F-actin mean in a shell shell_um outside the nucleus (excluding DAPI foreground), Ki67 mean inside.
    Class: neuron if MAP2 shell > map2_min, else astrocyte if F-actin shell > actin_min, else other; Ki67+ if Ki67 > ki67_min.
    MAP2 is measured in the shell, not as a shell/inside ratio: in 3D the axial blur fills the nuclear void with cytoplasm.
    Returns (N, 9): [z, y, x, r_um, class, ki67_pos, map2_shell, actin_shell, ki67_in], z/y/x in pixels of the full stack."""
    fxy = max(1, int(round(voxel_um / xy_res)))
    res = np.array([z_res, xy_res * fxy, xy_res * fxy])
    raw, zscore = {}, {}
    for name in ('dapi', 'map2', 'actin', 'ki67'):
        if name not in channels:
            continue
        raw[name] = downscale_local_mean(np.asarray(channels[name], dtype=np.float32), (1, fxy, fxy))
        bg_sub = ndimage.gaussian_filter(raw[name], 0.7 / res) - ndimage.gaussian_filter(raw[name], background_um / res)
        med = np.median(bg_sub)
        zscore[name] = (bg_sub - med) / (1.4826 * np.median(np.abs(bg_sub - med)))

    fg = zscore['dapi'] > filters.threshold_otsu(zscore['dapi'])
    blob = ndimage.gaussian_filter(raw['dapi'], 1.5 / res)  # one maximum per nucleus
    seeds = peak_local_max(blob, min_distance=int(round(2.5 / res[1])), labels=fg.astype(int), exclude_border=False)
    markers = np.zeros(fg.shape, int)
    markers[tuple(seeds.T)] = np.arange(1, len(seeds) + 1)
    nuclei = segmentation.watershed(-blob, markers, mask=fg)

    rows = []
    pad = np.ceil(shell_um[1] / res).astype(int) + 1
    for p in measure.regionprops(nuclei):
        r = (3 * p.area * res.prod() / (4 * np.pi)) ** (1 / 3)
        if r < min_radius_um:
            continue
        box = tuple(slice(max(0, a - q), b + q) for a, b, q in zip(p.bbox[:3], p.bbox[3:], pad))
        nuc = nuclei[box] == p.label
        dist = ndimage.distance_transform_edt(~nuc, sampling=res)
        shell = (dist > shell_um[0]) & (dist <= shell_um[1]) & ~fg[box]
        inner = ndimage.distance_transform_edt(nuc, sampling=res) > 1.0
        if inner.sum() < 5:
            inner = nuc
        mean = lambda name, region: zscore[name][box][region].mean() if name in zscore and region.any() else np.nan
        map2, actin, ki67 = mean('map2', shell), mean('actin', shell), mean('ki67', inner)
        cls = 0 if map2 > map2_min else 1 if actin > actin_min else 2
        z, y, x = p.centroid
        rows.append([z, (y + 0.5) * fxy - 0.5, (x + 0.5) * fxy - 0.5, r, cls, float(ki67 > ki67_min), map2, actin, ki67])
    return np.array(rows).reshape(-1, 9)


class ImageProcessor:
    def __init__(self, image_path = None, data_path='data/', channel=None, channel_names=None):

        self.data_path = data_path
        self.channel = channel
        self.channel_order = channel_names  # names of the TIFF channels in order (--channels); None: from the file name
        self.file_save = 'somas_data.npy'
        self.image_path = image_path
        self.channels = None  # {name: (Z, Y, X) array} for multi-channel TIFFs
        self.channel_names = None
        self.folder_save = os.path.join(data_path, image_path.split('.')[0])
        self.astrocytes_file = os.path.join(self.folder_save, 'astrocytes_data.npy')
        self.check = self.check_already_processed()
        self.read_image(image_path)
        self.data_folder()
        self.auto_somas = self.load_or_detect_somas()
        self.auto_cells = self.load_or_detect_cells() if self.channels else None

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
        print("Running automatic soma detection (background-subtracted triangle threshold + small-hole fill + opening + distance-transform peaks, r=8 um)...")
        somas = detect_somas(self.image_data, self.xy_res, self.z_res)
        np.save(auto_file, somas)
        print(f"Saved {len(somas)} automatic soma proposals to {auto_file}")
        return somas

    def load_or_detect_cells(self):
        """Load the DAPI/MAP2/F-actin/Ki67 cell proposals, or run detect_cells once and save them to auto_cells.npy."""
        auto_file = os.path.join(self.folder_save, 'auto_cells.npy')
        if os.path.exists(auto_file):
            cells = np.load(auto_file)
            print(f"Loaded {len(cells)} automatic cell proposals from {auto_file}")
        else:
            print("Running automatic cell detection (DAPI nuclei, classified by MAP2 / F-actin around them, Ki67 inside)...")
            cells = detect_cells(self.channels, self.xy_res, self.z_res)
            np.save(auto_file, cells)
            print(f"Saved {len(cells)} automatic cell proposals to {auto_file}")
        counts = ', '.join(f"{np.sum(cells[:, 4] == i)} {c}" for i, c in enumerate(CELL_CLASSES))
        print(f"  {counts}; {int(cells[:, 5].sum())} Ki67+")
        return cells

    def read_image(self, image_path):
        # Multi-channel TIFFs are re-read: only the soma channel is cached in processed_image.npy
        if self.check == True and not self.channel_names:
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
            self.channel_names = metadata.get('channel_names')
            if self.channel is None:
                self.channel = metadata.get('channel')
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
            series = tif.series[0]
            XYRes = tif.pages[0].tags.get("XResolution")
            self.xy_res = XYRes.value[1] / XYRes.value[0]  # µm/pixel
            # ImageJ omits "spacing=" when the voxel depth is 1.0
            self.z_res = float((tif.imagej_metadata or {}).get('spacing', 1.0))  # µm/pixel in Z
            image_data = series.asarray()

        if 'C' in series.axes:
            # Hyperstack (Z, C, Y, X): keep every channel, use one of them (MAP2 by default) for the somas
            image_data = np.moveaxis(image_data, series.axes.index('C'), 0)
            # Only channels named 'actin' / 'ki67' are used as F-actin / Ki67 by detect_cells; other names are display-only
            order = self.channel_order or channel_order_from_name(self.image_path)
            names = [order[c] if c < len(order) else f'ch{c + 1}' for c in range(len(image_data))]
            self.channels = dict(zip(names, image_data))
            self.channel_names = names
            if self.channel is None:
                self.channel = names.index(SOMA_CHANNEL)
            print(f"Channels: {', '.join(f'{i}={n}' for i, n in enumerate(names))}; somas on channel {self.channel} ({names[self.channel]})")
            image_data = image_data[self.channel]

        self.image_data = image_data
        self.z_px, self.height_px, self.width_px = image_data.shape

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
        self._add_ellipsoid(points[-1], self.xy_radius, self.z_radius)

        print(len(self.shapes_layer.data))

    def _add_ellipsoid(self, center_point, xy_radius, z_radius_max):
        """Add one 2D ellipse per z-slice (z = center +- z_radius_max) to the shapes layer: a soma ellipsoid."""
        z_radius = np.arange(-z_radius_max, z_radius_max+1)
        x_radius = [int(xy_radius * np.sqrt(1 - (z / z_radius_max) ** 2)) for z in z_radius]
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
        if self.channels:
            for name, data in self.channels.items():
                self.viewer.add_image(data, name=name, colormap=CHANNEL_COLORMAPS.get(name, 'gray'), blending='additive',
                                      visible=name in ('dapi', self.channel_names[self.channel]))
        else:
            self.viewer.add_image(self.image_data)
        self.cell_layers = {}  # 0 neuron, 1 astrocyte, 2 other, 3 proliferating -> editable points layer (select / move / delete with napari's own tools)
        single_channel = self.auto_cells is None
        if len(self.auto_somas):
            # Proposals only (not saved as somas); added before 'Soma Centroids' so clicks still go to the manual layer
            auto_layer = self.viewer.add_points(self.auto_somas[:, :3], name='Auto Somas (proposed)', ndim=3,
                                                size=2 * self.auto_somas[:, 3] / self.xy_res, face_color='transparent',
                                                border_color='yellow', opacity=0.8, visible=single_channel,
                                                out_of_slice_display=single_channel)
            if single_channel:
                # Without the multi-channel classes these are the neuron proposals: editable and accepted with 'Accept neuron somas'
                auto_layer.current_size = np.median(auto_layer.size)
                self.cell_layers[0] = auto_layer
        if self.auto_cells is not None:
            # One layer per group (DAPI nuclei): Ki67+ cells form their own group whatever their MAP2 / F-actin class;
            # hovering shows the class and the measured z-scores
            ki67 = self.auto_cells[:, 5] == 1
            groups = [('Neuron somas', 'yellow', (self.auto_cells[:, 4] == 0) & ~ki67),
                      ('Astrocyte somas', 'magenta', (self.auto_cells[:, 4] == 1) & ~ki67),
                      ('Other nuclei', 'white', (self.auto_cells[:, 4] == 2) & ~ki67),
                      ('Proliferating (Ki67+) cells', 'cyan', ki67)]
            for i, (name, color, selected) in enumerate(groups):
                cells = self.auto_cells[selected]
                kwargs = dict(name=f'{name} (proposed)', border_color=color,
                              features={'class': [CELL_CLASSES[int(k)] for k in cells[:, 4]], 'map2_shell': cells[:, 6],
                                        'actin_shell': cells[:, 7], 'ki67_in': cells[:, 8]})
                if i == 1 and os.path.exists(self.astrocytes_file):
                    cells = np.load(self.astrocytes_file)  # previously accepted astrocytes [z, y, x, r_um]
                    kwargs = dict(name=f'{name} (accepted)', border_color=color)
                if len(cells) == 0:
                    continue
                self.cell_layers[i] = self.viewer.add_points(cells[:, :3], ndim=3, size=2 * cells[:, 3] / self.xy_res,
                                                             face_color='transparent', border_width=0.15, opacity=0.8,
                                                             visible=i != 2, out_of_slice_display=True, **kwargs)
                self.cell_layers[i].current_size = np.median(self.cell_layers[i].size)  # size of points added by hand
        if self.cell_layers:
            # out_of_slice_display uses the size in pixels (~75) as z extent, i.e. every slice: limit it to the nucleus' slices
            self.viewer.dims.events.current_step.connect(self._show_cells_near_slice)
            self.viewer.dims.events.ndisplay.connect(self._show_cells_near_slice)
            self._show_cells_near_slice()

        self.points_layer = self.viewer.add_points(name='Soma Centroids', face_color='magenta', size=5, ndim=3)
        self.points_layer.events.data.connect(self._on_point_added)
        self.shapes_layer = self.viewer.add_shapes(name='Soma Ellipsoids', edge_color='red', face_color='red', ndim=3, opacity=0.5)
        self.viewer.window._qt_window.closeEvent = self._on_close
        #napari.view_image()
        self.viewer.window.add_dock_widget(self._slider_widget, name="Soma Controls", area='right')
        self.viewer.window.add_dock_widget(self._undo_widget, name="Undo Soma", area='right')
        self.viewer.window.add_dock_widget(self._save_widget, name="Save Somas", area='right')
        if self.cell_layers:
            actions = {'Accept neuron somas': self.accept_neurons, 'Accept astrocytes': self.accept_astrocytes,
                       'Discard other nuclei': self.discard_other_nuclei}
            if single_channel:
                actions = {'Accept neuron somas': self.accept_neurons}
            buttons = [PushButton(text=text) for text in actions]
            for button, action in zip(buttons, actions.values()):
                button.clicked.connect(action)
            self.viewer.window.add_dock_widget(Container(widgets=buttons), name="Cell Proposals", area='right')
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
            last = self.points_layer.data[-1]
            self.points_layer.remove(list(range(len(self.points_layer.data) - 1, len(self.points_layer.data))))
            # The soma's ellipses are the shapes centred on its (y, x)
            own = [i for i, c in enumerate(self.shapes_layer.data) if np.allclose(c[:, 1:].mean(axis=0), last[1:], atol=0.5)]
            self.shapes_layer.remove(own)
            #self.points_layer.events.data.connect(self._on_point_added)
        else:
            print("No soma points to undo.")

    def _show_cells_near_slice(self, event=None):
        """Show each cell proposal only on the slices its nucleus covers (centre +- radius in um / z_res); all of them in 3D."""
        z = self.viewer.dims.current_step[0]
        for layer in self.cell_layers.values():
            if self.viewer.dims.ndisplay == 3:
                layer.shown = np.ones(len(layer.data), bool)
            else:
                z_extent = np.asarray(layer.size) / 2 * self.xy_res / self.z_res
                layer.shown = np.abs(layer.data[:, 0] - z) <= z_extent

    def accept_neurons(self):
        """Copy the (edited) neuron proposals into 'Soma Centroids', each with an ellipsoid of NEURON_SEED_FRACTION times its point's
        radius (the detected nucleus radius; the median one for points added by hand), then remove the proposal layer.
        Proposals inside an existing soma (e.g. after Load Somas) are skipped. Save Somas writes them as usual."""
        layer = self.cell_layers.pop(0, None)
        if layer is None:
            print("No neuron proposals to accept.")
            return
        existing = np.array(self.points_layer.data)
        self.points_layer.events.data.disconnect(self._on_point_added)  # ellipsoids are added here with their own radius
        added = 0
        for center, size in zip(np.round(layer.data), layer.size):
            xy_r = NEURON_SEED_FRACTION * size / 2
            z_r = max(1, int(round(xy_r * self.xy_res / self.z_res)))
            if len(existing) and np.any((np.hypot(*(existing[:, 1:] - center[1:]).T) < xy_r) & (np.abs(existing[:, 0] - center[0]) <= z_r)):
                continue
            self.points_layer.add(center)
            self._add_ellipsoid(center, xy_r, z_r)
            added += 1
        self.points_layer.events.data.connect(self._on_point_added)
        self.viewer.layers.remove(layer)
        print(f"Accepted {added} neuron somas ({len(layer.data) - added} skipped: already marked). Press 'Save Somas' to save them.")

    def accept_astrocytes(self):
        """Save the (edited) astrocyte proposals to astrocytes_data.npy ([z, y, x, r_um], pixel coords); press again to re-save."""
        layer = self.cell_layers.get(1)
        if layer is None:
            print("No astrocyte proposals to accept.")
            return
        np.save(self.astrocytes_file, np.column_stack([layer.data, np.asarray(layer.size) / 2 * self.xy_res]))
        layer.name = 'Astrocyte somas (accepted)'
        print(f"Saved {len(layer.data)} astrocytes to {self.astrocytes_file}")

    def discard_other_nuclei(self):
        """Remove the 'other nuclei' proposals from the viewer (auto_cells.npy is kept)."""
        layer = self.cell_layers.pop(2, None)
        if layer is not None:
            self.viewer.layers.remove(layer)

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
            'channel_names': self.channel_names,
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
        default=None,
        help="Channel index (0-based) used for the somas: required for .nd2; multi-channel .tif default: MAP2 (index 1)."
    )
    parser.add_argument(
        "--channels",
        type=lambda s: [n.strip().lower() for n in s.split(',')],
        default=None,
        help="Comma-separated channel names of a multi-channel .tif in Fiji order, e.g. dapi,map2,piezo1,tuj1 "
             "(default from the file name: '_Mech_' in it -> dapi,map2,piezo1,tuj1; '_ID_' or neither -> "
             "dapi,map2,actin,ki67). Needs 'dapi' and 'map2'; "
             "only 'actin' / 'ki67' are used for the astrocyte / Ki67+ proposals."
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
            channel=args.channel,
            channel_names=args.channels
        )
        # Start the annotation process
        
        processor.open_viewer()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()