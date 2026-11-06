"""HTML viewer of the k-means step of the segmentation (intermediate result): raw channel | k-means clusters |
clusters kept by the curvature threshold | final segmentation, side by side, per z-slice and per segmented channel.

Two ways to get the cluster maps:
  - segment_neurons.py --save_kmeans: saved during a full run (whole stack), then make_kmeans_html is called from there.
  - this script: recomputes the k-means of a few size_lim blocks of an already segmented image (no full re-run):
        python kmeans_viewer.py data/<stem>/ --tag _sigma2_map2-tuj1 --sigma 2 --tile 1024 0 --z 25 40
    Each block is clustered independently (KMeans random_state=42), so a recomputed block equals the one of the full run.
"""
import os
import io
import json
import base64
import argparse
import time
import numpy as np
import tifffile as tiff
from PIL import Image
from NeuronSegmentation import SegmentationConfig, NeuronSegmenter


def block_stats(voxel, clusters):
    """Per-cluster statistics of one k-means block and its curvature threshold (as NeuronSegmenter.compute_neuron_mask):
    clusters >= thr are kept. Coordinates in the image (Z, Y, X) convention: y = segmenter axis 0, x = axis 1."""
    km, img = voxel['TOT_KM1'], voxel['reshaped_cIMc']
    G = [voxel['GxxKt'], voxel['GyyKt'], voxel['GzzKt']]
    stats, means = [], np.zeros((3, clusters))
    for c in range(clusters):
        sel = km == c
        n = int(sel.sum())
        if n:
            means[:, c] = [g[sel].mean() for g in G]
        stats.append({'n': n, 'I': float(img[sel].mean()) if n else 0.0,
                      'Gxx': float(means[0, c]), 'Gyy': float(means[1, c]), 'Gzz': float(means[2, c])})
    thr = int(NeuronSegmenter._mask_threshold(means[0], means[1], means[2]))
    nz = km.shape[2]
    return {'y0': int(voxel['nxi']), 'y1': int(voxel['nxi'] + km.shape[0]), 'x0': int(voxel['nyi']),
            'x1': int(voxel['nyi'] + km.shape[1]), 'z0': int(voxel['k']), 'z1': int(voxel['k'] + nz), 'thr': thr, 'stats': stats}


def kmeans_blocks(image, resolution, config, keys):
    """Recompute the level-1 k-means of the blocks `keys` ('crop_{y}_{x}_{z}', segmenter naming) of a (Z, Y, X) channel.
    config must be the one of the original run (same background_threshold). Returns {key: (TOT_KM1 (Y, X, Z), block_stats)}."""
    segmenter = NeuronSegmenter(config)
    segmenter.load_array(image, resolution)
    voxels = segmenter.voxelize_image()
    out = {}
    for key in keys:
        t = time.time()
        voxel = voxels[key]
        derivatives = segmenter.process_derivatives(voxel)
        segmenter.reshape_voxel(voxel, derivatives)
        voxel['TOT_KM1'], _ = segmenter.kmeans_clustering(voxel, derivatives)
        out[key] = (voxel['TOT_KM1'], block_stats(voxel, config.clusters))
        print(f"  {key}: threshold {out[key][1]['thr']} in {time.time() - t:.0f} s")
    return out


def _b64(array, fmt, **kwargs):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format=fmt, **kwargs)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _down(a, f, how):
    """Downsample the last two axes of (Z, Y, X) by f (block mean or max)."""
    if f == 1:
        return a
    nz, ny, nx = a.shape[0], a.shape[1] // f * f, a.shape[2] // f * f
    b = a[:, :ny, :nx].reshape(nz, ny // f, f, nx // f, f)
    return b.mean(axis=(2, 4)) if how == 'mean' else b.max(axis=(2, 4))


def make_kmeans_html(out_html, channels, seg, origin, xy_res, z_res, clusters, somas=None, downsample=1, title='K-means'):
    """Write the viewer. channels: list of dicts {name, raw (Z, Y, X), km (Z, Y, X) uint8 (0 = background), blocks (block_stats list),
    bg_threshold, intensity_ref}; seg (Z, Y, X) bool final segmentation of the same region; origin (z0, y0, x0) of the region in the
    image; somas (N, 3) (z, y, x) image pixels. Refuses to overwrite."""
    if os.path.exists(out_html):
        raise FileExistsError(f"Not overwriting {out_html}")
    assert clusters <= 8, "cluster ids are packed in 3 bits"
    seg_d = _down(seg.astype(np.uint8), downsample, 'max')
    chans = []
    for ch in channels:
        raw, km = ch['raw'], ch['km'].astype(np.uint8)
        kept = np.zeros(km.shape, bool)
        for b in ch['blocks']:
            sl = (slice(b['z0'] - origin[0], b['z1'] - origin[0]), slice(b['y0'] - origin[1], b['y1'] - origin[1]),
                  slice(b['x0'] - origin[2], b['x1'] - origin[2]))
            kept[sl] = km[sl] >= b['thr']
        lo, hi = np.percentile(raw[:, ::4, ::4], [0.5, 99.8])
        raw8 = (np.clip((_down(raw.astype(np.float32), downsample, 'mean') - lo) / max(hi - lo, 1), 0, 1) * 255).astype(np.uint8)
        packed = (_down(km, downsample, 'max') | (_down(kept.astype(np.uint8), downsample, 'max') << 3) | (seg_d << 4)).astype(np.uint8)
        packed *= 8  # values 8 apart: browsers may decode PNG gray levels off by 1 (colour management), JS rounds back
        chans.append({'name': ch['name'], 'lo': float(lo), 'hi': float(hi), 'bg_threshold': float(ch['bg_threshold']),
                      'intensity_ref': float(ch['intensity_ref']), 'blocks': ch['blocks'],
                      'raw': [_b64(s, 'JPEG', quality=88) for s in raw8], 'lab': [_b64(s, 'PNG', optimize=True) for s in packed]})
        print(f"  {ch['name']}: kept clusters = {(kept & (km > 0)).mean():.3f} of the region; final segmentation {seg.mean():.3f}")
    data = {'title': title, 'origin': [int(o) for o in origin], 'shape': [int(s) for s in seg.shape], 'ds': downsample,
            'xy_res': float(xy_res), 'z_res': float(z_res), 'clusters': clusters, 'channels': chans,
            'somas': [] if somas is None else np.asarray(somas, float).round(1).tolist()}
    with open(out_html, 'w') as f:
        f.write(HTML.replace('__TITLE__', title).replace('__DATA__', json.dumps(data)))
    print(f"Wrote {out_html} ({os.path.getsize(out_html) / 1e6:.1f} MB)")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#111316;--panel:#1a1d21;--line:#2c3036;--text:#e6e8eb;--muted:#9aa1aa;--accent:#ffb347;--seg:#ff8a3d}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 system-ui,-apple-system,Segoe UI,sans-serif}
header{display:flex;flex-wrap:wrap;gap:16px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5}
h1{font-size:15px;margin:0 12px 0 0;font-weight:600}
.ctl{display:flex;align-items:center;gap:6px;color:var(--muted)}
.ctl input[type=range]{width:220px}
.seg button{background:var(--panel);color:var(--text);border:1px solid var(--line);padding:4px 10px;cursor:pointer}
.seg button:first-child{border-radius:4px 0 0 4px}.seg button:last-child{border-radius:0 4px 4px 0}
.seg button.on{background:#3a4250;border-color:#596273}
#zval{font-variant-numeric:tabular-nums;min-width:110px;color:var(--text)}
main{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:12px;padding:12px 16px}
@media (max-width:900px){main{grid-template-columns:1fr}}
#grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
@media (max-width:700px){#grid{grid-template-columns:1fr}}
.pan{background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.pan h2{font-size:12px;font-weight:600;margin:0;padding:6px 10px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between}
.pan h2 span{color:var(--muted);font-weight:400}
.vp{position:relative;aspect-ratio:var(--ar);overflow:hidden;cursor:crosshair;background:#000;touch-action:none}
.vp canvas{position:absolute;left:0;top:0;transform-origin:0 0;image-rendering:pixelated}
.vp .xh{position:absolute;width:13px;height:13px;margin:-7px 0 0 -7px;border:1.5px solid #fff;border-radius:50%;pointer-events:none;mix-blend-mode:difference;display:none}
aside{display:flex;flex-direction:column;gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px}
.card h3{margin:0 0 6px;font-size:12px;font-weight:600}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:3px 4px;text-align:right;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:500}td:first-child,th:first-child{text-align:left}
tr.hov td{background:#2d333c}tr.iso td{outline:1px solid var(--accent)}
tbody tr{cursor:pointer}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-2px;margin-right:5px;border:1px solid #0006}
.kept{color:var(--accent);font-weight:600}
#hover{min-height:54px;color:var(--muted)}#hover b{color:var(--text);font-weight:600}
.note{color:var(--muted);font-size:12px}
</style></head><body>
<header>
 <h1>__TITLE__</h1>
 <div class="ctl">Channel <span class="seg" id="chsel"></span></div>
 <div class="ctl">z <input type="range" id="z" min="0" value="0"><span id="zval"></span></div>
 <div class="ctl"><label><input type="checkbox" id="somas" checked> soma centres</label></div>
 <div class="ctl"><button id="reset" class="seg" style="all:unset;cursor:pointer;color:var(--muted)">reset view (r)</button></div>
</header>
<main>
 <div id="grid"></div>
 <aside>
  <div class="card"><h3>Hover</h3><div id="hover">Move over an image.</div></div>
  <div class="card"><h3 id="bt">Clusters of this block</h3>
   <table><thead><tr><th>cluster</th><th>voxels</th><th>mean I</th><th>Gxx</th><th>Gyy</th><th>Gzz</th></tr></thead><tbody id="tb"></tbody></table>
   <p class="note">Clusters are sorted by mean intensity (0 = below background threshold, plus the dimmest cluster). A block keeps every
   cluster from the one after the last cluster whose mean Gxx, Gyy and Gzz are <i>not</i> all negative (SENPAI threshold); kept clusters are
   <span class="kept">highlighted</span>. Click a row to show only that cluster in panel 2.</p>
  </div>
  <div class="card note" id="info"></div>
 </aside>
</main>
<script>
const D=__DATA__;
const K=D.clusters, [NZ,H,W]=D.shape, ds=D.ds, Hd=Math.floor(H/ds), Wd=Math.floor(W/ds);
// ordinal palette for clusters 1..K-1 (dim -> bright), cluster 0 = background
const PAL=[[38,40,46],[59,82,139],[33,145,140],[94,201,98],[200,224,32],[253,231,37],[255,160,80],[255,90,90]];
const pal=c=>PAL[c===0?0:Math.round(1+(c-1)*(Math.min(K,8)-2)/Math.max(K-2,1))]||PAL[c];
const st={ch:0,z:Math.floor(NZ/2),iso:-1,s:1,tx:0,ty:0,hx:-1,hy:-1};
const PANELS=[['raw','Raw'],['km','K-means clusters'],['kept','Kept clusters (≥ threshold)'],['seg','Final segmentation']];
const grid=document.getElementById('grid');document.documentElement.style.setProperty('--ar',Wd+'/'+Hd);
const P={};
for(const [id,t] of PANELS){const d=document.createElement('div');d.className='pan';
 d.innerHTML=`<h2>${t}<span id="s_${id}"></span></h2><div class="vp"><canvas width="${Wd}" height="${Hd}"></canvas><div class="xh"></div></div>`;
 grid.appendChild(d);P[id]={vp:d.querySelector('.vp'),cv:d.querySelector('canvas'),xh:d.querySelector('.xh')};P[id].ctx=P[id].cv.getContext('2d');}
document.getElementById('s_seg').textContent=D.channels.length>1?'union of '+D.channels.map(c=>c.name).join(' + '):'';
const chsel=document.getElementById('chsel');
D.channels.forEach((c,i)=>{const b=document.createElement('button');b.textContent=c.name;b.onclick=()=>{st.ch=i;st.iso=-1;draw()};chsel.appendChild(b)});
const zs=document.getElementById('z');zs.max=NZ-1;zs.value=st.z;zs.oninput=()=>{st.z=+zs.value;draw()};
document.getElementById('somas').onchange=draw;
// decode cache
const cache={};
function load(src){return new Promise(r=>{const im=new Image();im.onload=()=>{const c=document.createElement('canvas');c.width=Wd;c.height=Hd;
 const x=c.getContext('2d');x.drawImage(im,0,0);r(x.getImageData(0,0,Wd,Hd).data)};im.src=src})}
async function slice(ch,z){const k=ch+'_'+z;if(!cache[k]){const C=D.channels[ch];
 const [r,l]=await Promise.all([load('data:image/jpeg;base64,'+C.raw[z]),load('data:image/png;base64,'+C.lab[z])]);
 const raw=new Uint8Array(Wd*Hd),lab=new Uint8Array(Wd*Hd);for(let i=0;i<Wd*Hd;i++){raw[i]=r[4*i];lab[i]=Math.round(l[4*i]/8)}cache[k]={raw,lab}}return cache[k]}
function blockAt(ch,z,y,x){return D.channels[ch].blocks.find(b=>z>=b.z0&&z<b.z1&&y>=b.y0&&y<b.y1&&x>=b.x0&&x<b.x1)}
let cur=null;
async function draw(){
 const {raw,lab}=await slice(st.ch,st.z);cur={raw,lab};
 [...chsel.children].forEach((b,i)=>b.classList.toggle('on',i===st.ch));
 const zi=D.origin[0]+st.z;document.getElementById('zval').textContent=`slice ${zi} (${(zi*D.z_res).toFixed(1)} µm)`;
 const n=Wd*Hd,im={};for(const id in P)im[id]=P[id].ctx.createImageData(Wd,Hd);
 for(let i=0;i<n;i++){const g=raw[i],v=lab[i],c=v&7,kp=(v>>3)&1,sg=(v>>4)&1,o=4*i;
  let a=im.raw.data;a[o]=a[o+1]=a[o+2]=g;a[o+3]=255;
  a=im.km.data;let q=pal(c);if(st.iso>=0&&c!==st.iso)q=[q[0]*.15,q[1]*.15,q[2]*.15];a[o]=q[0];a[o+1]=q[1];a[o+2]=q[2];a[o+3]=255;
  a=im.kept.data;const gd=g*.45;if(kp){q=pal(c);a[o]=q[0];a[o+1]=q[1];a[o+2]=q[2]}else{a[o]=a[o+1]=a[o+2]=gd}a[o+3]=255;
  a=im.seg.data;if(sg){a[o]=gd*.4+255*.6;a[o+1]=gd*.4+138*.6;a[o+2]=gd*.4+61*.6}else{a[o]=a[o+1]=a[o+2]=gd}a[o+3]=255}
 for(const id in P)P[id].ctx.putImageData(im[id],0,0);
 if(document.getElementById('somas').checked){const x=P.seg.ctx;x.strokeStyle='#4fc3f7';x.lineWidth=Math.max(1,1.5/st.s*Wd/P.seg.vp.clientWidth);
  for(const [sz,sy,sx] of D.somas){const dz=Math.abs(sz-zi);if(dz>3)continue;const yy=(sy-D.origin[1])/ds,xx=(sx-D.origin[2])/ds;
   if(yy<0||xx<0||yy>=Hd||xx>=Wd)continue;x.globalAlpha=1-dz/4;x.beginPath();x.arc(xx,yy,Math.max(3,12/ds),0,6.3);x.stroke()}x.globalAlpha=1}
 // kept fraction in header
 let nk=0,ns=0;for(let i=0;i<n;i++){nk+=(lab[i]>>3)&1;ns+=(lab[i]>>4)&1}
 document.getElementById('s_kept').textContent=(100*nk/n).toFixed(1)+'% of slice';document.getElementById('s_seg').textContent=
  (D.channels.length>1?'union '+D.channels.map(c=>c.name).join('+')+' · ':'')+(100*ns/n).toFixed(1)+'%';
 xf();hover();}
function xf(){for(const id in P){const p=P[id],k=p.vp.clientWidth/Wd;p.cv.style.width=Wd+'px';p.cv.style.height=Hd+'px';
 p.cv.style.transform=`translate(${st.tx}px,${st.ty}px) scale(${k*st.s})`}placeXh()}
function placeXh(){for(const id in P){const p=P[id],k=p.vp.clientWidth/Wd*st.s;if(st.hx<0){p.xh.style.display='none';continue}
 p.xh.style.display='block';p.xh.style.left=(st.tx+(st.hx+.5)*k)+'px';p.xh.style.top=(st.ty+(st.hy+.5)*k)+'px'}}
let drag=null;
for(const id in P){const vp=P[id].vp;
 vp.addEventListener('wheel',e=>{e.preventDefault();const r=vp.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,
  f=Math.exp(-e.deltaY*.0015),s2=Math.min(40,Math.max(1,st.s*f)),g=s2/st.s;st.tx=mx-(mx-st.tx)*g;st.ty=my-(my-st.ty)*g;st.s=s2;clamp();xf()},{passive:false});
 vp.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,tx:st.tx,ty:st.ty};vp.setPointerCapture(e.pointerId)});
 vp.addEventListener('pointerup',()=>drag=null);
 vp.addEventListener('pointermove',e=>{if(drag){st.tx=drag.tx+e.clientX-drag.x;st.ty=drag.ty+e.clientY-drag.y;clamp();xf()}
  const r=vp.getBoundingClientRect(),k=vp.clientWidth/Wd*st.s;st.hx=Math.floor((e.clientX-r.left-st.tx)/k);st.hy=Math.floor((e.clientY-r.top-st.ty)/k);
  if(st.hx<0||st.hy<0||st.hx>=Wd||st.hy>=Hd)st.hx=st.hy=-1;placeXh();hover()});
 vp.addEventListener('pointerleave',()=>{st.hx=st.hy=-1;placeXh();hover()});}
function clamp(){const w=P.raw.vp.clientWidth,h=P.raw.vp.clientHeight;st.tx=Math.min(0,Math.max(w-w*st.s,st.tx));st.ty=Math.min(0,Math.max(h-h*st.s,st.ty))}
document.getElementById('reset').onclick=()=>{st.s=1;st.tx=st.ty=0;xf()};
addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'&&e.target.type!=='range')return;
 if(e.key==='r'){st.s=1;st.tx=st.ty=0;xf()}if(e.key==='ArrowUp'||e.key==='ArrowRight'){if(st.z<NZ-1){st.z++;zs.value=st.z;draw()}e.preventDefault()}
 if(e.key==='ArrowDown'||e.key==='ArrowLeft'){if(st.z>0){st.z--;zs.value=st.z;draw()}e.preventDefault()}});
addEventListener('resize',xf);
let lastBlock=null;
function hover(){const C=D.channels[st.ch],zi=D.origin[0]+st.z,el=document.getElementById('hover');
 let y=D.origin[1]+Math.floor(H/2),x=D.origin[2]+Math.floor(W/2),c=-1;
 if(st.hx>=0&&cur){const i=st.hy*Wd+st.hx,v=cur.lab[i];y=D.origin[1]+st.hy*ds;x=D.origin[2]+st.hx*ds;c=v&7;
  const g=cur.raw[i],I=C.lo+g/255*(C.hi-C.lo);
  el.innerHTML=`(z, y, x) = <b>(${zi}, ${y}, ${x})</b> px · (${(x*D.xy_res).toFixed(1)}, ${(y*D.xy_res).toFixed(1)}) µm<br>`+
   `${C.name} ≈ <b>${I.toFixed(0)}</b>${ds>1?' (block mean)':''} · cluster <b><span class="sw" style="background:rgb(${pal(c)})"></span>${c}</b>`+
   ` · ${(v>>3)&1?'<span class="kept">kept</span>':'not kept'} · ${(v>>4)&1?'<b>in segmentation</b>':'not segmented'}`}
 else el.textContent='Move over an image.';
 const b=blockAt(st.ch,zi,y,x);if(!b)return;
 const tot=b.stats.reduce((s,q)=>s+q.n,0);
 document.getElementById('bt').textContent=`${C.name} · block y ${b.y0}–${b.y1}, x ${b.x0}–${b.x1}, z ${b.z0}–${b.z1-1} · threshold ${b.thr}`;
 document.getElementById('tb').innerHTML=b.stats.map((q,j)=>`<tr data-c="${j}" class="${j===c?'hov':''} ${j===st.iso?'iso':''}">`+
  `<td class="${j>=b.thr?'kept':''}"><span class="sw" style="background:rgb(${pal(j)})"></span>${j}${j>=b.thr?' ✓':''}</td>`+
  `<td>${(100*q.n/tot).toFixed(1)}%</td><td>${q.I.toFixed(0)}</td><td>${q.Gxx.toFixed(1)}</td><td>${q.Gyy.toFixed(1)}</td><td>${q.Gzz.toFixed(1)}</td></tr>`).join('');
 lastBlock=b}
document.getElementById('tb').onclick=e=>{const r=e.target.closest('tr');if(!r)return;const c=+r.dataset.c;st.iso=st.iso===c?-1:c;draw()};
document.getElementById('info').innerHTML=`Region (z, y, x) origin ${D.origin.join(', ')}, size ${NZ} × ${H} × ${W} px`+
 (ds>1?`, shown at ${ds}× downsampling (labels: block max)`:'')+`; ${D.xy_res.toFixed(4)} µm/px, z ${D.z_res} µm.<br>`+
 D.channels.map(c=>`${c.name}: background threshold ${c.bg_threshold.toFixed(0)}, intensity ref ${c.intensity_ref.toFixed(0)}, display ${c.lo.toFixed(0)}–${c.hi.toFixed(0)}`).join('<br>')+
 '<br>Scroll to zoom, drag to pan (all panels synced), ←/→ change slice.';
draw();
</script></body></html>
"""


def main():
    from segment_neurons import read_channel, sigma_tag
    parser = argparse.ArgumentParser(description="Recompute the k-means of a few blocks of a segmented image and write an HTML viewer.")
    parser.add_argument("data_folder")
    parser.add_argument("--sigma", type=float, nargs='+', default=[0], help="sigma_gaussian of the run (as segment_neurons --sigma)")
    parser.add_argument("--tag", default=None, help="tag of the run's outputs (default: from --sigma and the channels)")
    parser.add_argument("--tile", type=int, nargs=2, default=[0, 0], metavar=('Y0', 'X0'), help="block origin in pixels (multiple of size_lim)")
    parser.add_argument("--z", type=int, nargs=2, default=None, metavar=('Z0', 'Z1'), help="slices z0..z1-1 (rounded out to whole blocks)")
    parser.add_argument("--size-lim", type=int, nargs=3, default=[1024, 1024, 5])
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    folder = args.data_folder
    meta = np.load(os.path.join(folder, 'processed_metadata.npy'), allow_pickle=True).item()
    names = meta.get('channel_names') or []
    extra = ['tuj1'] if 'tuj1' in names else []
    tag = args.tag if args.tag is not None else sigma_tag(args.sigma) + ''.join('_map2-' + n for n in extra)
    map2 = np.load(os.path.join(folder, 'processed_image.npy'))
    nz, ny, nx = map2.shape
    resolution = (meta['xy_res'], meta['xy_res'], meta['z_res'])
    config = SegmentationConfig()  # same settings as segment_neurons.segment_neurons
    config.size_lim = [min(args.size_lim[0], nx), min(args.size_lim[1], ny), args.size_lim[2]]
    config.sigma_gaussian = list(args.sigma)
    if len(config.sigma_gaussian) > 1:
        print("Note: two scales; only the level-1 clusters are shown (the final segmentation ORs both levels).")
    win = config.size_lim[2]
    z0, z1 = args.z if args.z else (0, nz)
    zs = list(range(z0 // win * win, min(z1, nz), win))
    y0, x0 = args.tile
    keys = [f'crop_{y0}_{x0}_{z}' for z in zs]
    zr = (zs[0], min(zs[-1] + win, nz))
    yr, xr = (y0, min(y0 + config.size_lim[0], ny)), (x0, min(x0 + config.size_lim[1], nx))
    region = (slice(*zr), slice(*yr), slice(*xr))
    out = args.out or os.path.join(folder, f'kmeans_viewer{tag}_y{y0}_x{x0}_z{zr[0]}-{zr[1]}.html')
    if os.path.exists(out):
        raise FileExistsError(f"Not overwriting {out}")

    images = [('map2', map2)]
    tif = os.path.join(meta.get('data_path', 'data/'), 'imgs', meta['image_path'])
    images += [(n, read_channel(tif, names.index(n))) for n in extra]
    channels = []
    for name, image in images:
        # segment_neurons shares one config between channels: the background threshold of the first (MAP2) channel is reused
        print(f"{name}: {len(keys)} blocks")
        res = kmeans_blocks(image, resolution, config, keys)
        km = np.concatenate([res[k][0] for k in keys], axis=2).transpose(2, 0, 1)
        channels.append({'name': name, 'raw': image[region], 'km': km, 'blocks': [res[k][1] for k in keys],
                         'bg_threshold': config.background_threshold,
                         'intensity_ref': NeuronSegmenter(SegmentationConfig()).compute_intensity_reference(image)})
    seg = tiff.imread(os.path.join(folder, f'neuron_segmentation{tag}.tiff'))[region].astype(bool)
    somas = np.load(os.path.join(folder, 'somas_data.npy')) if os.path.exists(os.path.join(folder, 'somas_data.npy')) else None
    make_kmeans_html(out, channels, seg, (zr[0], yr[0], xr[0]), meta['xy_res'], meta['z_res'], config.clusters, somas,
                     title=f"K-means · {os.path.basename(os.path.normpath(folder))}")


if __name__ == "__main__":
    main()
