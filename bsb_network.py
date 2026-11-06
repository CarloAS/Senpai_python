"""Two BSB networks of the reconstructed neurons of one image, simulated with NEST, compared in one HTML page.

    BSB_IGNORE_PARALLEL=1 micromamba run -n vadis-nest python bsb_network.py data/<stem>/ --tag _sigma2_map2-tuj1

Run in the `vadis-nest` env (NEST 3.9 + bsb 6.0.7 + bsb-nest 6.0.7). Inputs: <data>/processed_metadata.npy and
<data>/morphologies<tag>/neuron_<k>.swc (NeuronSkeletonizer.py; soma = first node, um). Outputs in
results/network/<stem><tag>/ (refuses to overwrite): net_morph.hdf5, net_soma.hdf5, morphologies_centered/*.swc,
spikes.npz, summary.json, network_activity.html.

Both networks have the same cells (one per reconstructed neuron, at its SWC soma) and the same NEST model and input:
- morphology network ("morph"): each cell gets its own SWC (custom distributor ByNearestSoma); connections = BSB VoxelIntersection
  of the morphologies (one connection per overlapping pair and direction, autapses removed by the strategy wrapper).
- soma network ("soma"): no morphology; each unordered pair is connected (both directions, as morphology overlap is ~symmetric:
  92% of the Mech morphology connections are reciprocal) with probability p(d) = p0 exp(-d^2 / (2 sigma^2)) of the soma distance d
  (custom strategy DistanceDependent). sigma is fitted (maximum likelihood) to the pairs of the morphology network and p0 is set so that
  the expected number of connections equals the morphology network's: same count and same distance dependence, so differences in
  activity come from which pairs are connected. (Mech: p ~0.7 below 20 um, 0.5 at 20-30, 0.2 at 30-50, 0.05 at 50-80 um: Gaussian-like,
  an exponential fits badly.)
NEST: aeif_cond_alpha cells (adaptive exponential IF, NEST defaults except tau_syn_ex 2 ms (1 nS -> 0.8 mV EPSP) and spike-frequency
adaptation a 2 nS, b 80 pA, tau_w 500 ms), static excitatory synapses (same weight and delay in both), independent Poisson input to
every cell (default 2000 Hz x 1 nS: ~2.6 Hz without recurrence), spike recorder on all cells. Only excitatory neurons (no inhibitory
population is reconstructed). Without adaptation (iaf_cond_alpha) the excitatory network is bistable: ~3.5 Hz at w_rec 1 nS, runaway
~100-300 Hz from 1.5 nS on; adaptation lets network bursts terminate, as in cultures.
In vadis-nest `import h5py` must come before `import nest` (else NEST segfaults on import); done here.
"""
import h5py  # before bsb (HDF5 initialisation order, see CLAUDE.md)
import os
import sys
import json
import glob
import argparse
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from bsb import Scaffold, Configuration, config, options
from bsb.config import types
from bsb.connectivity.strategy import ConnectionStrategy
from bsb.connectivity.detailed.voxel_intersection import VoxelIntersection
from bsb.placement.distributor import MorphologyDistributor

CELL = 'neuron'


# ---------------------------------------------------------------- custom BSB components (referenced by import path in the config)
@config.node
class ByNearestSoma(MorphologyDistributor):
    """Give each placed cell the morphology whose soma (SWC root, um) is at its position."""
    somas = config.attr(type=types.ndarray(), required=True)
    names = config.attr(type=types.list(str), required=True)

    def distribute(self, positions, morphologies, context):
        index = {m.name: i for i, m in enumerate(morphologies)}
        d = cdist(positions, np.asarray(self.somas, dtype=float).reshape(-1, 3))
        nearest = d.argmin(axis=1)
        if len(positions) and d.min(axis=1).max() > 1e-3:
            raise ValueError(f"cell {d.min(axis=1).argmax()} is {d.min(axis=1).max():.3f} um from every SWC soma")
        return np.array([index[self.names[k]] for k in nearest], dtype=int)


def gaussian_p(d, p0, sigma):
    return p0 * np.exp(-d ** 2 / (2 * sigma ** 2))


@config.node
class DistanceDependent(ConnectionStrategy):
    """Connect cells with probability p0 * exp(-d^2 / (2 sigma_um^2)), d = soma distance in um, no autapses. symmetric: one draw
    per unordered pair, connected in both directions (pre and post must be the same population)."""
    p0 = config.attr(type=float, required=True)
    sigma_um = config.attr(type=float, required=True)
    symmetric = config.attr(type=bool, default=True)
    seed = config.attr(type=int, default=0)

    def connect(self, pre, post):
        rng = np.random.default_rng(self.seed)
        for from_ps in pre.placement:
            fp = from_ps.load_positions()
            for to_ps in post.placement:
                tp = to_ps.load_positions()
                d = cdist(fp, tp)
                hit = (rng.random(d.shape) < gaussian_p(d, self.p0, self.sigma_um)) & (d > 0)
                if self.symmetric:
                    hit = np.triu(hit, 1)
                    hit = hit | hit.T
                i, j = np.nonzero(hit)
                src, dst = np.full((len(i), 3), -1), np.full((len(i), 3), -1)
                src[:, 0], dst[:, 0] = i, j
                self.connect_cells(from_ps, to_ps, src, dst)


@config.node
class VoxelIntersectionNoAutapse(VoxelIntersection):
    """BSB VoxelIntersection without self-connections (a morphology always overlaps itself)."""

    def connect_cells(self, pre_set, post_set, src_locs, dest_locs, tag=None):
        if pre_set.cell_type is post_set.cell_type and len(src_locs):
            keep = src_locs[:, 0] != dest_locs[:, 0]
            src_locs, dest_locs = src_locs[keep], dest_locs[keep]
        super().connect_cells(pre_set, post_set, src_locs, dest_locs, tag=tag)


# ---------------------------------------------------------------- inputs
def load_neurons(data_folder, tag, out_dir):
    """Soma positions (um, x y z) and soma-centred copies of the SWC files (BSB morphologies are relative to the cell position)."""
    files = sorted(glob.glob(os.path.join(data_folder, f'morphologies{tag}', 'neuron_*.swc')),
                   key=lambda f: int(os.path.basename(f)[7:-4]))
    if not files:
        raise FileNotFoundError(f"no SWC files in {data_folder}morphologies{tag}/ (run NeuronSkeletonizer.py first)")
    cdir = os.path.join(out_dir, 'morphologies_centered')
    os.makedirs(cdir)
    somas, names, paths, swcs = [], [], [], []
    for f in files:
        a = np.loadtxt(f, comments='#', ndmin=2)
        somas.append(a[0, 2:5].copy())
        c = a.copy(); c[:, 2:5] -= a[0, 2:5]
        name = os.path.basename(f)[:-4]
        path = os.path.join(cdir, name + '.swc')
        np.savetxt(path, c, fmt=['%d', '%d', '%.3f', '%.3f', '%.3f', '%.3f', '%d'])
        names.append(name); paths.append(path); swcs.append(a)
    return np.array(somas), names, paths, swcs


def base_config(hdf5, extent):
    cfg = Configuration.default(storage=dict(engine='hdf5', root=hdf5))
    cfg.network.x, cfg.network.y, cfg.network.z = extent
    cfg.network.chunk_size = [float(e) + 1 for e in extent]  # one chunk: one connect() call, positions kept in order
    cfg.partitions.add('culture', thickness=extent[2])
    cfg.regions.add('dish', type='stack', children=['culture'])
    return cfg


def build_morph(hdf5, extent, somas, names, paths, voxels):
    cfg = base_config(hdf5, extent)
    for n, p in zip(names, paths):
        cfg.morphologies.append(dict(file=p, name=n))
    cfg.cell_types.add(CELL, spatial=dict(radius=5, count=len(somas), morphologies=[dict(names=names)]))
    cfg.placement.add('fixed', strategy='bsb.placement.FixedPositions', partitions=['culture'], cell_types=[CELL], positions=somas,
                      distribute=dict(morphologies=dict(strategy='bsb_network.ByNearestSoma', somas=somas, names=names)))
    cfg.connectivity.add('contacts', strategy='bsb_network.VoxelIntersectionNoAutapse', presynaptic=dict(cell_types=[CELL]),
                         postsynaptic=dict(cell_types=[CELL]), voxels_pre=voxels, voxels_post=voxels, contacts=1)
    sc = Scaffold(cfg)
    sc.compile()
    return sc


def build_soma(hdf5, extent, somas, p0, sigma_um, seed):
    cfg = base_config(hdf5, extent)
    cfg.cell_types.add(CELL, spatial=dict(radius=5, count=len(somas)))
    cfg.placement.add('fixed', strategy='bsb.placement.FixedPositions', partitions=['culture'], cell_types=[CELL], positions=somas)
    cfg.connectivity.add('distance', strategy='bsb_network.DistanceDependent', presynaptic=dict(cell_types=[CELL]),
                         postsynaptic=dict(cell_types=[CELL]), p0=p0, sigma_um=sigma_um, seed=seed)
    sc = Scaffold(cfg)
    sc.compile()
    return sc


def connections(sc, name):
    """(pre, post) cell indices (placement-set order) of a connectivity set."""
    pre, post = sc.get_connectivity_set(name).load_connections().as_globals().all()
    return np.c_[pre[:, 0], post[:, 0]].astype(int)


def fit_distance_rule(pos, pairs):
    """Gaussian rule for the symmetric soma network: sigma by maximum likelihood on the unordered pairs connected (in either direction)
    in the morphology network; p0 then set so that the expected number of directed connections (2 per pair) equals len(pairs)."""
    n = len(pos)
    d = cdist(pos, pos)[np.triu_indices(n, 1)]
    conn = np.zeros((n, n), bool); conn[pairs[:, 0], pairs[:, 1]] = True
    cc = (conn | conn.T)[np.triu_indices(n, 1)]

    def nll(x):
        p = np.clip(gaussian_p(d, 1 / (1 + np.exp(-x[0])), np.exp(x[1])), 1e-12, 1 - 1e-12)
        return -(np.log(p[cc]).sum() + np.log1p(-p[~cc]).sum())
    best = min((minimize(nll, [0.0, np.log(s0)], method='Nelder-Mead') for s0 in (10, 20, 40, 80)), key=lambda r: r.fun)
    sigma = float(np.exp(best.x[1]))
    p0 = (len(pairs) / 2) / gaussian_p(d, 1.0, sigma).sum()
    if p0 > 1:
        raise ValueError(f"cannot match {len(pairs)} connections with sigma {sigma:.1f} um (p0 = {p0:.2f} > 1)")
    return float(p0), sigma


# ---------------------------------------------------------------- NEST
def add_simulation(sc, conn_name, a, seed):
    name = f'sim_seed{seed}'  # one simulation entry per run (a config entry cannot be re-assigned)
    sc.configuration.simulations[name] = dict(
        simulator='nest', duration=a.duration, resolution=0.1, seed=seed,
        cell_models={CELL: dict(model='aeif_cond_alpha', constants=dict(t_ref=2.0, tau_syn_ex=a.tau_syn, a=a.adapt_a, b=a.adapt_b, tau_w=a.tau_w))},
        connection_models={conn_name: dict(synapse=dict(model='static_synapse', weight=a.w_rec, delay=a.delay))},
        devices=dict(
            background=dict(device='poisson_generator', rate=a.input_rate, weight=a.w_in, delay=0.1,
                            targetting=dict(strategy='cell_model', cell_models=[CELL])),
            spikes=dict(device='spike_recorder', delay=0.1, targetting=dict(strategy='cell_model', cell_models=[CELL]))))
    return name


def simulate(sc, conn_name, a, seed):
    res = sc.run_simulation(add_simulation(sc, conn_name, a, seed))
    tr = [st for st in res.block.segments[0].spiketrains if st.annotations.get('device') == 'spikes'][0]
    t = np.asarray(tr.magnitude, float)
    s = np.asarray(tr.array_annotations['senders'], int)
    n = len(sc.get_placement_set(CELL))
    s = s - 1  # cells are the first nodes created after ResetKernel (ids 1..n, placement-set order)
    assert s.min() >= 0 and s.max() < n, 'unexpected NEST node ids'
    return t, s


def activity_stats(t, s, n, duration, bin_ms=5.0):
    rates = np.bincount(s, minlength=n) / (duration / 1000.0)
    cv = []
    for i in range(n):
        ti = np.sort(t[s == i])
        if len(ti) > 3:
            isi = np.diff(ti); cv.append(isi.std() / isi.mean())
    counts = np.histogram(t, bins=np.arange(0, duration + bin_ms, bin_ms))[0]
    fano = counts.var() / counts.mean() if counts.mean() > 0 else np.nan
    # synchrony: variance of the population rate / mean variance of single-cell rates (Golomb 2007, chi^2), 5 ms bins
    per = np.stack([np.histogram(t[s == i], bins=np.arange(0, duration + bin_ms, bin_ms))[0] for i in range(n)])
    v_cells = per.var(axis=1).mean()
    chi = float(np.sqrt(per.mean(axis=0).var() / v_cells)) if v_cells > 0 else np.nan
    out = {'mean_rate_hz': float(rates.mean()), 'median_rate_hz': float(np.median(rates)), 'silent_cells': int((rates == 0).sum()),
           'cv_isi_median': float(np.median(cv)) if cv else np.nan, 'fano_pop_5ms': float(fano), 'synchrony_chi': chi,
           'spikes': int(len(t))}
    out.update(burst_stats(t, s, n, duration))
    return out


def detect_bursts(t, s, n, duration, bin_ms=10.0, frac=0.2, gap_ms=20.0):
    """Network bursts: runs of 10 ms bins in which >= frac of the cells spike, gaps < gap_ms merged. Returns [(start, end) ms]."""
    edges = np.arange(0, duration + bin_ms, bin_ms)
    b = np.digitize(t, edges) - 1
    active = np.zeros(len(edges) - 1)
    for k in range(len(active)):
        active[k] = len(np.unique(s[b == k]))
    on = active >= frac * n
    bursts, start = [], None
    for k, v in enumerate(on):
        if v and start is None:
            start = k
        if not v and start is not None:
            bursts.append([edges[start], edges[k]]); start = None
    if start is not None:
        bursts.append([edges[start], duration])
    merged = []
    for st, en in bursts:
        if merged and st - merged[-1][1] < gap_ms:
            merged[-1][1] = en
        else:
            merged.append([st, en])
    return merged


def burst_stats(t, s, n, duration):
    bursts = detect_bursts(t, s, n, duration)
    inb = np.zeros(len(t), bool)
    part = []
    for st, en in bursts:
        m = (t >= st) & (t < en); inb |= m; part.append(len(np.unique(s[m])) / n)
    ibi = np.diff([b[0] for b in bursts])
    return {'bursts_per_s': len(bursts) / (duration / 1000.0), 'burst_duration_ms': float(np.mean([e - b for b, e in bursts])) if bursts else 0.0,
            'burst_participation': float(np.mean(part)) if part else 0.0, 'spikes_in_bursts': float(inb.mean()) if len(t) else 0.0,
            'ibi_cv': float(ibi.std() / ibi.mean()) if len(ibi) > 2 else np.nan}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('data_folder')
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default=None, help='output folder (default results/network/<stem><tag>/)')
    ap.add_argument('--voxels', type=int, default=500, help='VoxelIntersection voxels per morphology (BSB default 50 is very coarse)')
    ap.add_argument('--duration', type=float, default=10000.0, help='ms')
    ap.add_argument('--input-rate', type=float, default=2000.0, help='Poisson input rate per cell (Hz)')
    ap.add_argument('--w-in', type=float, default=1.0, help='Poisson input weight (nS)')
    ap.add_argument('--w-rec', type=float, default=8.0, help='recurrent synapse weight (nS), both networks (bursting regime)')
    ap.add_argument('--repeats', type=int, default=5, help='input seeds (both) and soma-network realisations (soma) for mean +- SD')
    ap.add_argument('--tau-syn', type=float, default=2.0, help='excitatory synaptic time constant (ms)')
    ap.add_argument('--adapt-a', type=float, default=2.0, help='aeif subthreshold adaptation a (nS)')
    ap.add_argument('--adapt-b', type=float, default=80.0, help='aeif spike-triggered adaptation b (pA)')
    ap.add_argument('--tau-w', type=float, default=500.0, help='aeif adaptation time constant (ms)')
    ap.add_argument('--no-html', action='store_true')
    ap.add_argument('--delay', type=float, default=1.5, help='recurrent synaptic delay (ms)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--build-only', action='store_true')
    a = ap.parse_args()
    options.verbosity = 1

    stem = os.path.basename(os.path.normpath(a.data_folder))
    out = a.out or os.path.join('results', 'network', stem + a.tag)
    if os.path.exists(out):
        raise FileExistsError(f"Not overwriting {out}")
    os.makedirs(out)
    meta = np.load(os.path.join(a.data_folder, 'processed_metadata.npy'), allow_pickle=True).item()
    extent = [meta['width_px'] * meta['xy_res'], meta['height_px'] * meta['xy_res'], meta['z_px'] * meta['z_res']]
    somas, names, paths, swcs = load_neurons(a.data_folder, a.tag, out)
    print(f"{len(somas)} neurons, network {extent[0]:.0f} x {extent[1]:.0f} x {extent[2]:.0f} um")

    morph = build_morph(os.path.join(out, 'net_morph.hdf5'), extent, somas, names, paths, a.voxels)
    pos = morph.get_placement_set(CELL).load_positions()
    order = cdist(pos, somas).argmin(axis=1)               # placement index -> SWC index
    assert len(set(order)) == len(somas)
    ms = morph.get_placement_set(CELL).load_morphologies()
    got = [ms.names[k] for k in ms.get_indices()]
    bad = sum(got[i] != names[order[i]] for i in range(len(pos)))
    if bad:
        raise RuntimeError(f"{bad} cells got a morphology that is not theirs")
    print(f"morphology check: all {len(pos)} cells carry their own SWC")
    pm = connections(morph, 'contacts')
    p0, sigma = fit_distance_rule(pos, pm)
    print(f"morphology network: {len(pm)} connections ({len(pm) / len(pos):.1f} per cell); soma rule p(d) = {p0:.3f} exp(-d^2 / 2 {sigma:.1f}^2)")
    soma = build_soma(os.path.join(out, 'net_soma.hdf5'), extent, pos, p0, sigma, a.seed)
    ps_ = connections(soma, 'distance')
    print(f"soma network: {len(ps_)} connections ({len(ps_) / len(pos):.1f} per cell)")
    summary = {'image': stem, 'tag': a.tag, 'neurons': len(pos), 'extent_um': extent, 'voxels': a.voxels,
               'fit': {'p0': p0, 'sigma_um': sigma}, 'params': vars(a),
               'morph': {'connections': int(len(pm))}, 'soma': {'connections': int(len(ps_))}}
    np.savez(os.path.join(out, 'network.npz'), positions=pos, swc_index=order, conn_morph=pm, conn_soma=ps_)
    if a.build_only:
        json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
        return
    # repeat r: input seed seed+r for both; the soma network is also re-drawn (connection seed seed+r). Run 0 is shown in the HTML.
    spikes, runs = {}, {'morph': [], 'soma': []}
    for r in range(a.repeats):
        soma_r = soma if r == 0 else build_soma(os.path.join(out, f'net_soma_r{r}.hdf5'), extent, pos, p0, sigma, a.seed + r)
        for key, sc, cn in (('morph', morph, 'contacts'), ('soma', soma_r, 'distance')):
            t, s = simulate(sc, cn, a, a.seed + r)
            if r == 0:
                spikes[key] = (t, s)
            st = activity_stats(t, s, len(pos), a.duration)
            if key == 'soma':
                st['connections'] = int(len(connections(soma_r, 'distance')))
            runs[key].append(st)
        print(f"run {r}: " + ' | '.join(f"{k} {runs[k][-1]['mean_rate_hz']:.2f} Hz, chi {runs[k][-1]['synchrony_chi']:.3f}, "
                                         f"{runs[k][-1]['bursts_per_s']:.2f} bursts/s" for k in runs))
    for key in runs:
        summary[key]['runs'] = runs[key]
        summary[key]['mean'] = {m: float(np.nanmean([x[m] for x in runs[key]])) for m in runs[key][0]}
        summary[key]['sd'] = {m: float(np.nanstd([x[m] for x in runs[key]], ddof=1)) if len(runs[key]) > 1 else 0.0 for m in runs[key][0]}
    np.savez(os.path.join(out, 'spikes.npz'), **{f'{k}_t': v[0] for k, v in spikes.items()}, **{f'{k}_s': v[1] for k, v in spikes.items()})
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    if a.no_html:
        return
    from network_html import write_html
    write_html(os.path.join(out, 'network_activity.html'), summary, pos, [swcs[k] for k in order], pm, ps_, spikes, a.duration)


if __name__ == '__main__':
    main()
