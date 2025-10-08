#%%
import bsb 
from bsb import Scaffold, Configuration
from NeuronalSomaMarker import *

somas = NeuronalSomaMarker(image_path="test_asia.tif", init_marks="test_asia_somas_mask.tif")
#somas.start_gui()
centers = somas.get_soma_centers()
#%%
cell_positions = centers
cell_num = len(cell_positions)

bsb.options.verbosity = 3
config = Configuration.default(storage=dict(engine="hdf5", root="network.hdf5"))
config.network.x = 480.40
config.network.y = 480.40
config.network.z = 39.0

config.partitions.add("cell_layer", thickness=39.0)
config.regions.add("brain_on_chip", type="stack", children=["cell_layer"])
config.cell_types.add(
  "cell_test",
  spatial=dict(radius=6, count = cell_num)
)
#%%
config.placement.add(
  "place_in_fixed_position",
  strategy="bsb.placement.FixedPositions",
  partitions=["cell_layer"],
  cell_types=["cell_test"],
  positions=cell_positions
)
#%%
config.connectivity.add(
  "test_to_test",
  strategy="bsb.connectivity.FixedIndegree",
  presynaptic=dict(cell_types=["cell_test"]),
  postsynaptic=dict(cell_types=["cell_test"]),
  indegree = int(cell_num*0.5),
)
# %%
