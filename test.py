from NeuronSegmentation import SegmentationConfig, NeuronSegmenter
from NeuronSeparator import *
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pickle as pkl
# Initialize segmentation

config = SegmentationConfig()
config.size_lim = [1024, 1024, 5]
config.sigma_gaussian = 0

segmenter = NeuronSegmenter(config)
orig_img = segmenter.load_image('results/test_output','test_neuron_1.tif')
voxels = segmenter.k_means_voxels()

TOT_KM1 = segmenter.recompose(voxels)

#segmenter.visualize_kmeans_with_slider(TOT_KM1[0])

seg_1, seg_2, refined_seg_1, refined_seg_2 = segmenter.process_segmentation(voxels)

pkl.dump(seg_1, open('results/test_output/seg_1.pkl', 'wb'))
pkl.dump(seg_2, open('results/test_output/seg_2.pkl', 'wb'))
pkl.dump(refined_seg_1, open('results/test_output/refined_seg_1.pkl', 'wb'))
pkl.dump(refined_seg_2, open('results/test_output/refined_seg_2.pkl', 'wb'))

segmenter.visualize_segmentation_comparison_xyz(orig_img, refined_seg_1)

separator = NeuronSeparator()
parcellated_neurons = separator.separate_neurons(refined_seg_1,orig_img,"test_neurons_1_somas_mask.tif")

visualize_parcellated_neurons_3d(parcellated_neurons)