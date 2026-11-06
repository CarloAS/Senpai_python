"""SWC morphologies of the separated neurons (replaces SENPAI senpai_skeletonize.m; see deviations below).

    python NeuronSkeletonizer.py data/<stem>/ --tag _sigma2_map2-tuj1

Reads neuron_labels<tag>.tiff, soma_mask.tiff and processed_metadata.npy of a folder written by Image_Processor.py /
segment_neurons.py; writes morphologies<tag>/neuron_<label>.swc (one per neuron, um) and morphologies<tag>/summary.csv.
Refuses to overwrite.

Per neuron, as MATLAB: neuron | soma, keep the 26-connected component of the soma, 3D hole fill, distance transform (here in um),
tree rooted at one soma node at the soma (seed) centroid, short terminal spurs dropped, radius = distance to the boundary.
Skeleton: TEASAR (Sato et al. 2000, as in kimimaro) instead of MATLAB's bwskel thinning + cycle cut. Reason: the 3D thinning of
the installed skimage (0.25.2, Lee, the bwskel equivalent) deletes every object whose cross-section is an even square (2x2, 4x4,
... voxels: a 6x6x50 box gives an empty skeleton), so neurites break or vanish. TEASAR: penalised shortest paths from the soma
through the mask (edge cost = length * (1 + 5000 (1 - DBF/DBFmax)^16), i.e. along the centre line); repeatedly trace the path
from the farthest (geodesic) remaining voxel to the tree and remove a rolling ball of radius scale*DBF + const around it.
The result is a tree (no cycles to cut); intensity is not used (MATLAB used it only for cycle cuts).
Other deviations: coordinates and radii in um (MATLAB: pixels; BSB needs um), x = image column, y = row, z = slice; the root is
always the soma; standard SWC types 1 soma / 3 neurite (MATLAB 0 / 5 / 6 / 1); the mask is binned in XY to about voxel_um (any).
"""
import os
import argparse
import numpy as np
import tifffile as tiff
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

OFFSETS = np.array([(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dz, dy, dx) > (0, 0, 0)])


def _bin_xy(a, b):
    """Bin the Y, X axes of a (Z, Y, X) bool array by b (any), padding with False."""
    if b == 1:
        return a
    nz, ny, nx = a.shape
    pad = np.zeros((nz, -(-ny // b) * b, -(-nx // b) * b), dtype=bool)
    pad[:, :ny, :nx] = a
    return pad.reshape(nz, pad.shape[1] // b, b, pad.shape[2] // b, b).any(axis=(2, 4))


def xy_bin_factor(xy_res, voxel_um=0.5):
    return max(1, int(round(voxel_um / xy_res)))


def teasar(neuron, soma, spacing, scale=3.0, const=1.0, min_branch_um=1.5):
    """TEASAR tree of one neuron. neuron, soma: (Z, Y, X) bool; spacing (dz, dy, dx) um.
    Returns coords (N, 3) voxel (z, y, x) (node 0 = soma centroid), radius (N,) um, parent (N,) (-1 for the soma); parents precede children."""
    neuron = neuron | soma
    lab, n = ndimage.label(neuron, structure=np.ones((3, 3, 3)))
    if n > 1:  # MATLAB keeps the largest component; prefer the one containing the soma
        counts = np.bincount(lab[soma], minlength=n + 1) if soma.any() else np.bincount(lab.ravel())
        counts[0] = 0
        neuron = lab == counts.argmax()
    neuron = ndimage.binary_fill_holes(neuron)
    dbf_vol = ndimage.distance_transform_edt(np.pad(neuron, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]  # crop border = outside
    soma = soma & neuron
    if not soma.any():  # no seed inside this label: the thickest point is the soma
        soma = np.zeros_like(neuron)
        soma[np.unravel_index(np.argmax(dbf_vol), dbf_vol.shape)] = True
    soma_c, soma_r = np.argwhere(soma).mean(0), float(dbf_vol[soma].max())

    vox = np.argwhere(neuron)
    nv = len(vox)
    index = np.full(neuron.shape, -1, dtype=np.int64)
    index[tuple(vox.T)] = np.arange(nv)
    dbf = dbf_vol[tuple(vox.T)]
    pen = 1.0 + 5000.0 * (1.0 - dbf / dbf.max()) ** 16
    ii, jj, ll = [], [], []
    for off in OFFSETS:
        q = vox + off
        ok = np.all((q >= 0) & (q < neuron.shape), axis=1)
        nb = np.full(nv, -1)
        nb[ok] = index[tuple(q[ok].T)]
        m = nb >= 0
        ii.append(np.nonzero(m)[0]); jj.append(nb[m]); ll.append(np.full(m.sum(), np.linalg.norm(off * spacing)))
    ii, jj, ll = np.concatenate(ii), np.concatenate(jj), np.concatenate(ll)
    # single source (the seed voxel deepest in the cell): all paths follow one shortest-path tree and merge where they meet
    # (with every seed voxel as a source, paths along the same neurite can run side by side to different seed voxels)
    seed_vox = np.argwhere(soma)
    sources = index[tuple(seed_vox[np.argmax(dbf_vol[tuple(seed_vox.T)])])]
    geo = dijkstra(coo_matrix((ll, (ii, jj)), shape=(nv, nv)).tocsr(), directed=False, indices=sources, min_only=True)
    W = coo_matrix((ll * (pen[ii] + pen[jj]) / 2, (ii, jj)), shape=(nv, nv)).tocsr()  # penalised: prefers the centre line

    is_source = np.zeros(nv, bool); is_source[sources] = True
    valid = np.isfinite(geo) & (geo > soma_r + const)  # voxels still to cover (the cell body around the seed is covered by the soma)
    node_of = np.full(nv, -1)                          # voxel -> tree node
    coords, radius, parent = [soma_c], [soma_r], [-1]
    vol_valid = np.zeros(neuron.shape, bool); vol_valid[tuple(vox[valid].T)] = True
    rmax = scale * dbf.max() + const
    halo = np.ceil(rmax / np.asarray(spacing)).astype(int)
    while vol_valid.any():
        cand = np.flatnonzero(vol_valid[tuple(vox.T)])
        v = cand[np.argmax(geo[cand])]
        # cheapest (penalised) path from the target to any voxel already in the tree (the source = soma, or a node):
        # new branches join the tree where they meet it (kimimaro fix_branching)
        cost, pred = dijkstra(W, directed=False, indices=v, return_predecessors=True)
        in_tree = np.flatnonzero((node_of >= 0) | is_source)
        dest = in_tree[np.argmin(cost[in_tree])]
        path, u = [], pred[dest]
        while u >= 0 and u != v:
            path.append(u); u = pred[u]
        path = [v] + path[::-1]  # target ... last voxel before the tree
        attach = node_of[dest] if node_of[dest] >= 0 else 0
        inside = np.flatnonzero(geo[path] <= soma_r)
        if len(inside):  # the path enters the cell body: cut it there and attach to the soma node (soma = one node, as MATLAB)
            path, attach = path[:inside[0]], 0
        if not path:
            continue
        pts = vox[path]
        length = np.linalg.norm(np.diff(pts, axis=0) * spacing, axis=1).sum()
        gap = np.linalg.norm((pts[-1] - coords[attach]) * spacing)
        length += max(0.0, gap - soma_r) if attach == 0 else gap  # stems count from the soma surface
        if length >= min_branch_um:
            for p in reversed(path):                       # from the attachment point outwards: parents precede children
                node_of[p] = len(coords)
                coords.append(vox[p].astype(float)); radius.append(float(dbf[p])); parent.append(attach)
                attach = node_of[p]
        # rolling ball: cover voxels within scale * DBF + const of the path (also when the path was too short to keep)
        lo = np.maximum(pts.min(0) - halo, 0); hi = np.minimum(pts.max(0) + halo + 1, neuron.shape)
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        pm = np.ones(hi - lo, bool); pm[tuple((pts - lo).T)] = False
        d, ind = ndimage.distance_transform_edt(pm, sampling=spacing, return_indices=True)
        ball = d <= scale * dbf_vol[box][tuple(ind)] + const
        vol_valid[box] &= ~ball
    return prune_spurs(np.array(coords), np.array(radius), np.array(parent), spacing, min_branch_um)


def prune_spurs(coords, radius, parent, spacing, min_branch_um):
    """Drop terminal pieces shorter than min_branch_um that end at a branch point (as MATLAB bwskel MinBranchLength); one pass.
    Needed for loops, where TEASAR traces the second half in short steps that leave 1-node spurs."""
    n = len(parent)
    nchild = np.bincount(parent[1:], minlength=n)
    seglen = np.r_[0, np.linalg.norm((coords[1:] - coords[parent[1:]]) * spacing, axis=1)]
    keep = np.ones(n, bool)
    for leaf in np.flatnonzero(nchild == 0):
        if leaf == 0:
            continue
        piece, length, v = [leaf], seglen[leaf], parent[leaf]
        while v > 0 and nchild[v] == 1:
            piece.append(v); length += seglen[v]; v = parent[v]
        if v > 0 and nchild[v] > 1 and length < min_branch_um:
            keep[piece] = False
    new = np.cumsum(keep) - 1
    return coords[keep], radius[keep], np.where(parent[keep] >= 0, new[np.maximum(parent[keep], 0)], -1)


def skeletonize_crop(lab, soma, xy_res, z_res, voxel_um=0.5, **kw):
    """teasar on a (Z, Y, X) crop at native resolution, XY-binned to about voxel_um; coords returned in native crop pixels (z, y, x)."""
    b = xy_bin_factor(xy_res, voxel_um)
    coords, radius, parent = teasar(_bin_xy(lab, b), _bin_xy(soma, b), (z_res, xy_res * b, xy_res * b), **kw)
    px = np.c_[coords[:, 0], coords[:, 1] * b + (b - 1) / 2, coords[:, 2] * b + (b - 1) / 2]
    return px, radius, parent


def write_swc(path, coords_um, radius, parent, header=''):
    """SWC: id type x y z radius parent (ids from 1; type 1 soma, 3 neurite)."""
    with open(path, 'w') as f:
        f.write(header)
        for i, ((x, y, z), r, p) in enumerate(zip(coords_um, radius, parent)):
            f.write(f"{i + 1} {1 if p < 0 else 3} {x:.3f} {y:.3f} {z:.3f} {r:.3f} {p + 1 if p >= 0 else -1}\n")


def tree_stats(coords_um, parent):
    seg = np.linalg.norm(coords_um[1:] - coords_um[parent[1:]], axis=1)
    nchild = np.bincount(parent[1:], minlength=len(parent))
    reach = np.linalg.norm(coords_um - coords_um[0], axis=1).max()
    return {'nodes': len(parent), 'length_um': float(seg.sum()), 'stems': int(nchild[0]), 'branch_points': int((nchild[1:] > 1).sum()),
            'tips': int((nchild[1:] == 0).sum()), 'max_reach_um': float(reach)}


def skeletonize_folder(data_folder, tag='', voxel_um=0.5, margin=2, **kw):
    meta = np.load(os.path.join(data_folder, 'processed_metadata.npy'), allow_pickle=True).item()
    xy, zr = meta['xy_res'], meta['z_res']
    b = xy_bin_factor(xy, voxel_um)
    out_dir = os.path.join(data_folder, f'morphologies{tag}')
    if os.path.exists(out_dir):
        raise FileExistsError(f"Not overwriting {out_dir}")
    labels = tiff.imread(os.path.join(data_folder, f'neuron_labels{tag}.tiff'))
    somas = tiff.imread(os.path.join(data_folder, 'soma_mask.tiff')) > 0
    print(f"{labels.max()} labels, XY bin {b} -> grid {xy * b:.3f} x {xy * b:.3f} x {zr:.3f} um")
    os.makedirs(out_dir)
    rows = []
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        sl = tuple(slice(max(0, s.start - margin), min(n, s.stop + margin)) for s, n in zip(sl, labels.shape))
        lab = labels[sl] == k
        px, radius, parent = skeletonize_crop(lab, somas[sl] & lab, xy, zr, voxel_um, **kw)
        xyz_um = (px + [s.start for s in sl])[:, ::-1] * [xy, xy, zr]  # crop -> image pixels -> um; SWC order x, y, z
        header = (f"# skeleton (TEASAR) of neuron label {k} from {data_folder} (neuron_labels{tag}.tiff)\n"
                  f"# units um; x = column * {xy:.6f}, y = row * {xy:.6f}, z = slice * {zr:.6f}; XY bin {b}; type 1 soma, 3 neurite\n")
        write_swc(os.path.join(out_dir, f'neuron_{k}.swc'), xyz_um, radius, parent, header)
        st = tree_stats(xyz_um, parent)
        rows.append({'label': k, **st})
        print(f"  neuron {k}: {st['nodes']} nodes, {st['length_um']:.0f} um, {st['stems']} stems, {st['branch_points']} branch points")
    with open(os.path.join(out_dir, 'summary.csv'), 'w') as f:
        f.write(','.join(rows[0]) + '\n')
        for row in rows:
            f.write(','.join(f'{v:.2f}' if isinstance(v, float) else str(v) for v in row.values()) + '\n')
    L = np.array([row['length_um'] for row in rows])
    print(f"Wrote {len(rows)} SWC files to {out_dir}; total length median {np.median(L):.0f} um (5-95%: "
          f"{np.percentile(L, 5):.0f}-{np.percentile(L, 95):.0f})")
    return out_dir


def main():
    p = argparse.ArgumentParser(description="SWC skeletons (um) of the neurons separated by segment_neurons.py.")
    p.add_argument("data_folder")
    p.add_argument("--tag", default='', help="segment_neurons output tag, e.g. _sigma2_map2-tuj1")
    p.add_argument("--voxel-um", type=float, default=0.5, help="XY binning target, um (default 0.5)")
    p.add_argument("--scale", type=float, default=3.0, help="TEASAR rolling ball: radius = scale * DBF + const (default 3; 1.5 leaves many spurs in blobby cell bodies)")
    p.add_argument("--const", type=float, default=1.0, help="TEASAR rolling ball constant, um (default 1)")
    p.add_argument("--min-branch-um", type=float, default=1.5, help="drop new branches shorter than this, um (default 1.5)")
    a = p.parse_args()
    skeletonize_folder(a.data_folder, a.tag, a.voxel_um, scale=a.scale, const=a.const, min_branch_um=a.min_branch_um)


if __name__ == "__main__":
    main()
