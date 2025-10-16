import bsb
from bsb import Scaffold, Configuration
import numpy as np
import os

class BSB_compile:
    def __init__(self, somas_file=None, morph_file=None, output_file="new_network.hdf5"):
        self.somas_file = somas_file
        self.morph_file = morph_file
        self.output_file = output_file
        self.data_path = 'data/'
        self.cell_densities = 2.22*10**(-6)  # cells per um^3
        self.load_data()

    def load_data(self):
        """Load .npy for metadata, image_data and somas coordinates.""" 
        try:
            self.metadata = np.load(os.path.join(self.data_path, "processed_metadata.npy"), allow_pickle=True)
            print(f"Data loaded from processed_metadata.npy")
            self.load_metadata()
            
        except Exception as e:
            print(f"Error loading processed_metadata.npy: {e}")

        try:
            self.image_data = np.load(os.path.join(self.data_path, "processed_image.npy"), allow_pickle=True)
            print(f"Data loaded from processed_image.npy")
        except Exception as e:
            print(f"Error loading processed_image.npy: {e}")

    def load_metadata(self):
        """Extract metadata values from the loaded metadata dictionary."""
        self.width_px = self.metadata.item().get('width_px')
        self.height_px = self.metadata.item().get('height_px')
        self.z_px = self.metadata.item().get('z_px')
        self.xy_res = self.metadata.item().get('xy_res')  # in um/pixel
        self.z_res = self.metadata.item().get('z_res')    # in um/pixel
        print(f"Metadata extracted: width={self.width_px}, height={self.height_px}, z={self.z_px}, xy_res={self.xy_res}, z_res={self.z_res}")

    def load_somas(self):
        """Load soma coordinates from .npy file."""
        if self.somas_file:
            try:
                self.cell_positions = np.load(self.somas_file)
                print(f"Soma coordinates loaded from {self.somas_file}")
                self.placement_strategy = "FixedPositions"
            except Exception as e:
                print(f"Error loading soma coordinates from {self.somas_file}: {e}")
                self.placement_strategy = "Random"
        else:
            self.cell_positions = np.empty((0, 3))
            print("No soma file provided, using empty array.")

    def load_morphologies(self):
        """Load neuron morphologies if provided."""
        if self.morph_file:
            try:
                self.morphologies = np.load(self.morph_file, allow_pickle=True)
                print(f"Morphologies loaded from {self.morph_file}")
                self.conn_strategy = "VoxelIntersection"
            except Exception as e:
                print(f"Error loading morphologies from {self.morph_file}: {e}")
                self.conn_strategy = "FixedIndegree"
        else:
            self.morphologies = None
            print("No morphology file provided.")

    def initialize_network(self, conn_strategy=None):
        """Initialize the BSB network configuration.\nProvide conn_strategy to override default:\n--- AlltoAll, FixedIndegree, FixedOutdegree, VoxelIntersection ---"""
        
        bsb.options.verbosity = 3
        config = Configuration.default(storage=dict(engine="hdf5", root=self.output_file))
        config.network.x = self.width_px * self.xy_res
        config.network.y = self.height_px * self.xy_res
        config.network.z = self.z_px * self.z_res
        self.load_somas

        config.partitions.add("cell_layer", thickness=config.network.z)

        if self.placement_strategy == "Random":
            volume = config.network.x * config.network.y * config.network.z  # in um^3
            estimated_cells = int(volume * self.cell_densities)
            self.cell_num = estimated_cells
        else:
            self.cell_num = self.cell_positions.shape[0]
        
        config.regions.add("brain_on_chip", type="stack", children=["cell_layer"])
        config.cell_types.add(
            "cell_test",
            spatial=dict(radius=6, count=self.cell_num)
        )

        self.load_morphologies()
        self.conn_strategy = conn_strategy or self.conn_strategy
        print(f"Initialized network with: \n{self.cell_num} cells\nPlacement strategy: {self.placement_strategy}\nConnectivity strategy: {self.conn_strategy}")

        self.config = config
        


    def config_network(self):

        if self.placement_strategy == "Random":
            self.config.placement.add(
                "place_randomly",
                strategy=f"bsb.placement.{self.placement_strategy}",
                partitions=["cell_layer"],
                cell_types=["cell_test"]
            )
        else:
            self.config.placement.add(
                "place_in_fixed_position",
                strategy=f"bsb.placement.{self.placement_strategy}",
                partitions=["cell_layer"],
                cell_types=["cell_test"],
                positions=self.cell_positions
            )

        cases = {
            "AlltoAll": dict(),
            "FixedIndegree": dict(indegree=int(self.cell_num * 0.5)),
            "FixedOutdegree": dict(outdegree=int(self.cell_num * 0.5)),
            "VoxelIntersection": dict(morphologies=self.morphologies)
        }  
        conn_params = cases.get(self.conn_strategy, {})
        if not conn_params:
            print(f"Warning: Unknown connectivity strategy '{self.conn_strategy}'. Defaulting to 'FixedIndegree'.")
            conn_params = cases["FixedIndegree"]

        
        self.config.connectivity.add(
            "test_to_test",
            strategy=f"bsb.connectivity.{self.conn_strategy}",
            presynaptic=dict(cell_types=["cell_test"]),
            postsynaptic=dict(cell_types=["cell_test"]),
            **conn_params
        )

    def build_and_save(self):
        """Build and save the network to the specified HDF5 file."""
        scaffold = Scaffold(self.config)
        scaffold.build()
        scaffold.save()

    def plot_network(self):
        """Visualize the network using BSB's built-in plotting."""
        scaffold = Scaffold(self.config)
        scaffold.build()
        scaffold.plot_3d()