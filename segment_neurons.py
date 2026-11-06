import os
import argparse
import time
import numpy as np
import tifffile as tiff
from NeuronSegmentation import SegmentationConfig, NeuronSegmenter
from NeuronSeparator import NeuronSeparator


def sigma_tag(sigma_gaussian):
    """File name tag of a sigma_gaussian setting: '' for the default [0], else e.g. '_sigma2' or '_sigma0-3'."""
    sigma = list(np.atleast_1d(sigma_gaussian))
    return '' if sigma == [0] else '_sigma' + '-'.join(f'{s:g}' for s in sigma)


def segment_neurons(data_folder, size_lim=(1024, 1024, 5), sigma_gaussian=(0,), soma_grow_um=6.0):
    """Segment the soma channel (MAP2) of an image processed with Image_Processor.py and separate it into single neurons.
    Reads <data_folder>/processed_image.npy (soma channel, (Z, Y, X)), processed_metadata.npy and soma_mask.tiff (one seed
    per accepted soma); writes neuron_segmentation.tiff (bool) and neuron_labels.tiff (uint16, one label per soma seed),
    both (Z, Y, X) like the input; file names carry sigma_tag(sigma_gaussian). sigma_gaussian (pixels, as MATLAB sig_G): [0] as
    test.py; with two scales the two refined segmentations are OR-ed, as MATLAB. The soma seeds are grown by up to soma_grow_um inside the
    cell body before the watershed (NeuronSeparator, resolution given). Existing outputs are never overwritten; an existing
    neuron_segmentation.tiff is reused (only the separation is run: delete it to segment again)."""
    tag = sigma_tag(sigma_gaussian)
    out_seg = os.path.join(data_folder, f'neuron_segmentation{tag}.tiff')
    out_lab = os.path.join(data_folder, f'neuron_labels{tag}.tiff')
    if os.path.exists(out_lab):
        raise FileExistsError(f"Not overwriting {out_lab}")

    metadata = np.load(os.path.join(data_folder, 'processed_metadata.npy'), allow_pickle=True).item()
    image = np.load(os.path.join(data_folder, 'processed_image.npy'))
    print(f"Image {image.shape}, {metadata['xy_res']:.4f} um (XY), {metadata['z_res']:.4f} um (Z)")

    config = SegmentationConfig()
    # Crop windows larger than the image break voxelize_image's crop sizes (NxC = window, not image, size)
    config.size_lim = [min(size_lim[0], image.shape[2]), min(size_lim[1], image.shape[1]), size_lim[2]]
    config.sigma_gaussian = list(np.atleast_1d(sigma_gaussian))
    segmenter = NeuronSegmenter(config)
    img = segmenter.load_array(image, (metadata['xy_res'], metadata['xy_res'], metadata['z_res']))

    if os.path.exists(out_seg):
        refined_seg_1 = tiff.imread(out_seg).transpose(1, 2, 0).astype(bool)
        print(f"Reusing {out_seg}: {refined_seg_1.sum()} voxels ({refined_seg_1.mean():.3f} of the stack)")
    else:
        t = time.time()
        voxels = segmenter.k_means_voxels()
        _, _, refined_seg_1, refined_seg_2 = segmenter.process_segmentation(voxels)
        refined_seg_1 = np.asarray(refined_seg_1, dtype=bool)
        if len(config.sigma_gaussian) > 1:
            refined_seg_1 |= np.asarray(refined_seg_2, dtype=bool)  # MATLAB: one segmentation from both scales
        print(f"Segmentation: {refined_seg_1.sum()} voxels ({refined_seg_1.mean():.3f} of the stack) in {time.time() - t:.0f} s")
        tiff.imwrite(out_seg, refined_seg_1.transpose(2, 0, 1))

    t = time.time()
    labels = NeuronSeparator().separate_neurons(refined_seg_1, img, os.path.join(data_folder, 'soma_mask.tiff'),
                                                resolution=segmenter.image_info['resolution'], soma_grow_um=soma_grow_um)
    print(f"Separation: {len(np.unique(labels)) - 1} neurons in {time.time() - t:.0f} s")
    tiff.imwrite(out_lab, labels.transpose(2, 0, 1))
    print(f"Saved {out_seg} and {out_lab}")
    return refined_seg_1, labels


def main():
    parser = argparse.ArgumentParser(description="Segment and separate neurons for an image processed with Image_Processor.py.")
    parser.add_argument("data_folder", help="Folder written by Image_Processor.py, e.g. data/IMAGE_STEM/")
    parser.add_argument("--sigma", type=float, nargs='+', default=[0],
                        help="Gaussian sigma(s) in pixels before the derivatives (MATLAB sig_G): 0 (default), e.g. 2, or two scales 0 3.")
    parser.add_argument("--soma-grow-um", type=float, default=6.0, help="Maximum growth of the soma seeds in um (default 6).")
    args = parser.parse_args()
    segment_neurons(args.data_folder, sigma_gaussian=args.sigma, soma_grow_um=args.soma_grow_um)


if __name__ == "__main__":
    main()
