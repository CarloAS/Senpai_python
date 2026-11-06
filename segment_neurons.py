import os
import json
import argparse
import time
import numpy as np
import tifffile as tiff
from NeuronSegmentation import SegmentationConfig, NeuronSegmenter
from NeuronSeparator import NeuronSeparator
from kmeans_viewer import block_stats, make_kmeans_html


def sigma_tag(sigma_gaussian):
    """File name tag of a sigma_gaussian setting: '' for the default [0], else e.g. '_sigma2' or '_sigma0-3'."""
    sigma = list(np.atleast_1d(sigma_gaussian))
    return '' if sigma == [0] else '_sigma' + '-'.join(f'{s:g}' for s in sigma)


def read_channel(tif_path, channel):
    """One channel (Z, Y, X) of an ImageJ hyperstack (pages in Z, C order) without reading the other channels."""
    with tiff.TiffFile(tif_path) as t:
        series = t.series[0]
        n_c, n_z = series.shape[series.axes.index('C')], series.shape[series.axes.index('Z')]
        return t.asarray(key=range(channel, n_z * n_c, n_c))


def segment_channel(image, resolution, config, keep_kmeans=False):
    """Topology-informed k-means segmentation of one (Z, Y, X) channel; returns the (Y, X, Z) bool segmentation and the segmenter.
    keep_kmeans: also return (level-1 k-means labels (Y, X, Z) uint8, per-block cluster stats and threshold, see kmeans_viewer.block_stats)."""
    segmenter = NeuronSegmenter(config)
    segmenter.load_array(image, resolution)
    voxels = segmenter.k_means_voxels()
    km_lv1, _, refined_seg_1, refined_seg_2 = segmenter.process_segmentation(voxels)
    seg = np.asarray(refined_seg_1, dtype=bool)
    if len(config.sigma_gaussian) > 1:
        seg |= np.asarray(refined_seg_2, dtype=bool)  # MATLAB: one segmentation from both scales
    if keep_kmeans:
        return seg, segmenter, (km_lv1, [block_stats(v, config.clusters) for v in voxels.values()])
    return seg, segmenter


def segment_neurons(data_folder, size_lim=(1024, 1024, 5), sigma_gaussian=(0,), soma_grow_um=6.0, save_kmeans=False,
                    fill_z_support=None):
    """Segment the neuronal channels of an image processed with Image_Processor.py and separate them into single neurons.
    Segmented channels: MAP2 (processed_image.npy); if the image has a 'tuj1' channel (Mech stain) also Tuj1, read from the original TIFF
    (<data_path>/imgs/<image_path> of the metadata), segmented separately with the same settings and OR-ed with MAP2 (file tag '_map2-tuj1').
    The separation (soma seed growth + watershed) always runs on MAP2.
    Reads <data_folder>/processed_image.npy (soma channel, (Z, Y, X)), processed_metadata.npy and soma_mask.tiff (one seed
    per accepted soma); writes neuron_segmentation.tiff (bool) and neuron_labels.tiff (uint16, one label per soma seed),
    both (Z, Y, X) like the input; file names carry sigma_tag(sigma_gaussian). sigma_gaussian (pixels, as MATLAB sig_G): [0] as
    test.py; with two scales the two refined segmentations are OR-ed, as MATLAB. The soma seeds are grown by up to soma_grow_um inside the
    cell body before the watershed (NeuronSeparator, resolution given). Existing outputs are never overwritten; an existing
    neuron_segmentation.tiff is reused (only the separation is run: delete it to segment again).
    save_kmeans: also write the level-1 k-means labels of each segmented channel (kmeans_<channel><tag>.tiff, (Z, Y, X) uint8,
    0..clusters-1 by ascending intensity), their per-block stats/thresholds (kmeans_blocks<tag>.json) and the viewer
    kmeans_viewer<tag>.html (2x downsampled); needs a new segmentation (fails if neuron_segmentation<tag>.tiff exists).
    fill_z_support: SegmentationConfig.fill_z_support (z-consistent hole filling, e.g. 0.5; None = MATLAB); adds '_zfill<f>' to the tag."""
    metadata = np.load(os.path.join(data_folder, 'processed_metadata.npy'), allow_pickle=True).item()
    names = metadata.get('channel_names') or []
    extra = ['tuj1'] if 'tuj1' in names else []
    tag = sigma_tag(sigma_gaussian) + (f'_zfill{fill_z_support:g}' if fill_z_support is not None else '') + ''.join('_map2-' + n for n in extra)
    out_seg = os.path.join(data_folder, f'neuron_segmentation{tag}.tiff')
    out_lab = os.path.join(data_folder, f'neuron_labels{tag}.tiff')
    if os.path.exists(out_lab):
        raise FileExistsError(f"Not overwriting {out_lab}")
    km_files = [os.path.join(data_folder, f'kmeans_{n}{tag}.tiff') for n in ['map2'] + extra]
    km_files += [os.path.join(data_folder, f'kmeans_blocks{tag}.json'), os.path.join(data_folder, f'kmeans_viewer{tag}.html')]
    if save_kmeans:
        if os.path.exists(out_seg):
            raise FileExistsError(f"--save_kmeans needs a new segmentation, but {out_seg} exists (it would be reused)")
        for f in km_files:
            if os.path.exists(f):
                raise FileExistsError(f"Not overwriting {f}")

    image = np.load(os.path.join(data_folder, 'processed_image.npy'))
    print(f"Image {image.shape}, {metadata['xy_res']:.4f} um (XY), {metadata['z_res']:.4f} um (Z)")

    config = SegmentationConfig()
    # Crop windows larger than the image break voxelize_image's crop sizes (NxC = window, not image, size)
    config.size_lim = [min(size_lim[0], image.shape[2]), min(size_lim[1], image.shape[1]), size_lim[2]]
    config.sigma_gaussian = list(np.atleast_1d(sigma_gaussian))
    config.fill_z_support = fill_z_support
    resolution = (metadata['xy_res'], metadata['xy_res'], metadata['z_res'])
    segmenter = NeuronSegmenter(config)
    img = segmenter.load_array(image, resolution)  # MAP2 (Y, X, Z): separation image

    if os.path.exists(out_seg):
        refined_seg_1 = tiff.imread(out_seg).transpose(1, 2, 0).astype(bool)
        print(f"Reusing {out_seg}: {refined_seg_1.sum()} voxels ({refined_seg_1.mean():.3f} of the stack)")
    else:
        t = time.time()
        kmeans = []  # (name, raw (Z, Y, X), (km (Y, X, Z), blocks)) per channel, only with save_kmeans
        refined_seg_1, _, *km = segment_channel(image, resolution, config, keep_kmeans=save_kmeans)
        kmeans += [('map2', image, km[0])] if km else []
        print(f"Segmentation (map2): {refined_seg_1.sum()} voxels ({refined_seg_1.mean():.3f} of the stack) in {time.time() - t:.0f} s")
        tif_path = os.path.join(metadata.get('data_path', 'data/'), 'imgs', metadata['image_path'])
        for name in extra:
            t = time.time()
            chan = read_channel(tif_path, names.index(name))
            seg, _, *km = segment_channel(chan, resolution, config, keep_kmeans=save_kmeans)
            kmeans += [(name, chan, km[0])] if km else []
            print(f"Segmentation ({name}): {seg.sum()} voxels ({seg.mean():.3f}); only in {name}: {(seg & ~refined_seg_1).sum()} "
                  f"voxels, in {time.time() - t:.0f} s")
            refined_seg_1 |= seg
        if extra:
            print(f"Segmentation (union): {refined_seg_1.sum()} voxels ({refined_seg_1.mean():.3f} of the stack)")
        tiff.imwrite(out_seg, refined_seg_1.transpose(2, 0, 1))
        if save_kmeans:
            save_kmeans_outputs(kmeans, refined_seg_1, km_files, config, metadata, data_folder)

    t = time.time()
    labels = NeuronSeparator().separate_neurons(refined_seg_1, img, os.path.join(data_folder, 'soma_mask.tiff'),
                                                resolution=segmenter.image_info['resolution'], soma_grow_um=soma_grow_um)
    print(f"Separation: {len(np.unique(labels)) - 1} neurons in {time.time() - t:.0f} s")
    tiff.imwrite(out_lab, labels.transpose(2, 0, 1))
    print(f"Saved {out_seg} and {out_lab}")
    return refined_seg_1, labels


def save_kmeans_outputs(kmeans, seg, km_files, config, metadata, data_folder):
    """Write the k-means label TIFFs, the per-block stats JSON and the k-means viewer (see segment_neurons save_kmeans)."""
    channels = []
    for (name, raw, (km, blocks)), f in zip(kmeans, km_files):
        tiff.imwrite(f, km.transpose(2, 0, 1))
        channels.append({'name': name, 'raw': raw, 'km': km.transpose(2, 0, 1), 'blocks': blocks,
                         'bg_threshold': config.background_threshold,  # shared config: the MAP2 threshold is used for every channel
                         'intensity_ref': NeuronSegmenter(SegmentationConfig()).compute_intensity_reference(raw)})
    with open(km_files[-2], 'w') as fh:
        json.dump({c['name']: c['blocks'] for c in channels}, fh)
    somas_file = os.path.join(data_folder, 'somas_data.npy')
    make_kmeans_html(km_files[-1], channels, seg.transpose(2, 0, 1), (0, 0, 0), metadata['xy_res'], metadata['z_res'], config.clusters,
                     np.load(somas_file) if os.path.exists(somas_file) else None, downsample=2,
                     title=f"K-means · {os.path.basename(os.path.normpath(data_folder))}")
    print(f"Saved {', '.join(km_files)}")


def main():
    parser = argparse.ArgumentParser(description="Segment and separate neurons for an image processed with Image_Processor.py.")
    parser.add_argument("data_folder", help="Folder written by Image_Processor.py, e.g. data/IMAGE_STEM/")
    parser.add_argument("--sigma", type=float, nargs='+', default=[0],
                        help="Gaussian sigma(s) in pixels before the derivatives (MATLAB sig_G): 0 (default), e.g. 2, or two scales 0 3.")
    parser.add_argument("--soma-grow-um", type=float, default=6.0, help="Maximum growth of the soma seeds in um (default 6).")
    parser.add_argument("--save_kmeans", action="store_true",
                        help="Also save the k-means labels (kmeans_<channel><tag>.tiff, kmeans_blocks<tag>.json) and kmeans_viewer<tag>.html.")
    parser.add_argument("--fill-z-support", type=float, default=None,
                        help="z-consistent hole filling: fill a slice's hole only if >= this fraction of it is filled in an adjacent "
                             "slice (e.g. 0.5; default: off, MATLAB behaviour). Adds _zfill<f> to the output tag.")
    args = parser.parse_args()
    segment_neurons(args.data_folder, sigma_gaussian=args.sigma, soma_grow_um=args.soma_grow_um, save_kmeans=args.save_kmeans,
                    fill_z_support=args.fill_z_support)


if __name__ == "__main__":
    main()
