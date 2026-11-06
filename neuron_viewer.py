import os
import io
import json
import base64
import argparse
import numpy as np
import tifffile as tiff
from PIL import Image


def _b64(array, fmt, **kwargs):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format=fmt, **kwargs)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _block_max(a, f):
    """Downsample (Y, X) by f taking the block maximum, so thin labelled neurites stay visible."""
    ny, nx = a.shape[0] // f * f, a.shape[1] // f * f
    return a[:ny, :nx].reshape(ny // f, f, nx // f, f).max(axis=(1, 3))


def _to_8bit(a, lo, hi):
    return (np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _surface_cells(lab3):
    """Cells of a (Z, Y, X) label grid with at least one 6-neighbour of another label (or outside the grid)."""
    pad = np.pad(lab3, 1)
    inner = np.ones(lab3.shape, bool)
    for ax in range(3):
        for d in (-1, 1):
            inner &= np.roll(pad, d, axis=ax)[1:-1, 1:-1, 1:-1] == lab3
    return (lab3 > 0) & ~inner


def _shading(lab3, surface, grid, sigma_um=1.0, ao_um=3.0):
    """Per surface cell (order of np.nonzero(surface)): outward normal of its own label, from the gradient of the label's mask
    smoothed by sigma_um, in the viewer's frame (x right, y up = -row, z up = -slice), as uint8 (n + 1) * 127.5; and an
    ambient-occlusion factor (uint8, 255 = open) from the fraction of labelled cells within ~ao_um (crevices, contacts darker)."""
    from scipy import ndimage
    normals = np.zeros(surface.shape + (3,), np.float32)
    pad = int(np.ceil(3 * sigma_um / grid.min())) + 1
    for k, box in enumerate(ndimage.find_objects(lab3), start=1):
        if box is None:
            continue
        box = tuple(slice(max(0, b.start - pad), b.stop + pad) for b in box)
        own = lab3[box] == k
        smooth = ndimage.gaussian_filter(own.astype(np.float32), sigma_um / grid)
        g = np.gradient(smooth, *grid)  # d/dz, d/dy, d/dx (um)
        n = np.stack([-g[2], g[1], g[0]], -1)  # outward = -gradient; viewer y = -row, z = -slice
        sel = own & surface[box]
        normals[box][sel] = n[sel]
    occ = ndimage.gaussian_filter((lab3 > 0).astype(np.float32), ao_um / grid)
    n = normals[surface]
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(length > 1e-6, n / np.maximum(length, 1e-6), [0, 0, 1])  # isolated cell: facing up
    ao = np.clip(1.25 - 1.4 * occ[surface], 0.3, 1.0)
    return np.round((n + 1) * 127.5).astype(np.uint8), np.round(ao * 255).astype(np.uint8)


def make_viewer(data_folder, out_html=None, downsample=2, dapi=None, tag='', voxel3d_um=0.7, max_points=1_500_000):
    """Write a self-contained HTML viewer of a separated stack. 2D: every z-slice of the soma channel (MAP2), optionally DAPI in
    blue, with neuron_labels.tiff as a coloured overlay; max-projection mode; hover = neuron id, click = isolate a neuron.
    3D (WebGL, no external library): every labelled voxel on a ~voxel3d_um XY grid (block maximum, all z), coloured per neuron,
    plus the unassigned segmentation (neuron_segmentation<tag>.tiff, if present) in grey; synced with the 2D selection. Only the surface
    cells of each label are drawn, lit with per-cell normals of the label's smoothed mask and darkened by ambient occlusion
    (_shading); a 'shading' checkbox switches back to flat colours.
    The 3D grid is made coarser (x2 in XY and z per step) until it has at most max_points surface points, to keep the page responsive.
    Reads <data_folder>/processed_image.npy, processed_metadata.npy, neuron_labels.tiff and somas_data.npy (if present).
    dapi: optional (Z, Y, X) array of the DAPI channel. 2D images are downsampled by `downsample` in XY (labels by block maximum).
    tag: file name tag of segment_neurons outputs (e.g. '_sigma2'): reads neuron_labels<tag>.tiff, writes neuron_viewer<tag>.html."""
    out_html = out_html or os.path.join(data_folder, f'neuron_viewer{tag}.html')
    if os.path.exists(out_html):
        raise FileExistsError(f"Not overwriting {out_html}")
    meta = np.load(os.path.join(data_folder, 'processed_metadata.npy'), allow_pickle=True).item()
    image = np.load(os.path.join(data_folder, 'processed_image.npy'))
    labels = tiff.imread(os.path.join(data_folder, f'neuron_labels{tag}.tiff'))
    if labels.max() > 65535 or labels.shape != image.shape:
        raise ValueError(f"labels {labels.shape} do not match the image {image.shape}")
    xy, z_res = meta['xy_res'], meta['z_res']

    # Neuron table from the full-resolution labels
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    neurons = []
    for k, n in zip(ids, counts):
        zz, yy, xx = np.nonzero(labels == k)
        neurons.append({'id': int(k), 'voxels': int(n), 'volume_um3': round(float(n * xy * xy * z_res), 1),
                        'z': [int(zz.min()), int(zz.max())], 'centre': [round(float(yy.mean()) / downsample, 1), round(float(xx.mean()) / downsample, 1)],
                        'extent_um': round(float(max(np.ptp(yy), np.ptp(xx)) * xy), 1)})

    lo, hi = np.percentile(image, [1, 99.7])
    dlo, dhi = np.percentile(dapi, [1, 99.7]) if dapi is not None else (0, 1)

    def rgb(z_img, z_dapi):
        """Soma channel in gray, or green with DAPI in blue."""
        g = _to_8bit(z_img[::downsample, ::downsample], lo, hi)
        if z_dapi is None:
            return np.stack([g, g, g], -1)
        return np.stack([np.zeros_like(g), g, _to_8bit(z_dapi[::downsample, ::downsample], dlo, dhi)], -1)

    def label_png(lab2d):
        lab2d = _block_max(lab2d, downsample).astype(np.uint16)
        return _b64(np.stack([lab2d // 256, lab2d % 256, np.zeros_like(lab2d)], -1).astype(np.uint8), 'PNG', optimize=True)

    slices = []
    for z in range(image.shape[0]):
        slices.append({'img': _b64(rgb(image[z], None), 'JPEG', quality=85),
                       'dapi': _b64(rgb(image[z], dapi[z]), 'JPEG', quality=85) if dapi is not None else None,
                       'lab': label_png(labels[z])})
    # Projection: max intensity; label = the label with most voxels along z at each pixel
    proj_lab = np.zeros(labels.shape[1:], np.uint16)
    best = np.zeros(labels.shape[1:], np.int32)
    for k in ids:
        c = (labels == k).sum(0)
        better = c > best
        proj_lab[better], best[better] = k, c[better]
    projection = {'img': _b64(rgb(image.max(0), None), 'JPEG', quality=85),
                  'dapi': _b64(rgb(image.max(0), dapi.max(0)), 'JPEG', quality=85) if dapi is not None else None,
                  'lab': label_png(proj_lab)}

    # 3D point cloud: [x, y, z, label] uint16 per point on the coarse grid; label 0 = segmented but not assigned to a neuron
    seg_file = os.path.join(data_folder, f'neuron_segmentation{tag}.tiff')
    seg = tiff.imread(seg_file).astype(np.uint8) if os.path.exists(seg_file) else None
    f3, fz = max(1, int(round(voxel3d_um / xy))), 1
    while True:
        nz = labels.shape[0] // fz * fz
        zmax = lambda a: a[:nz].reshape(nz // fz, fz, *a.shape[1:]).max(axis=1)
        lab3 = zmax(np.stack([_block_max(l, f3) for l in labels]))
        if seg is not None:
            lab3 = np.where((lab3 == 0) & (zmax(np.stack([_block_max(s, f3) for s in seg])) > 0), 65535, lab3)
        surface = _surface_cells(lab3)  # only the outer shell of each neuron is drawn (the inside is hidden)
        if np.count_nonzero(surface) <= max_points:
            break
        f3, fz = f3 * 2, fz * 2
    grid = np.array([z_res * fz, xy * f3, xy * f3])  # um per 3D cell (z, y, x)
    zz, yy, xx = np.nonzero(surface)
    ll = lab3[zz, yy, xx].astype(np.int64)
    normals, ao = _shading(lab3, surface, grid)
    ll[ll == 65535] = 0
    points = np.column_stack([xx, yy, zz, ll]).astype('<u2')
    points3d = base64.b64encode(points.tobytes()).decode('ascii')
    shade3d = base64.b64encode(np.column_stack([normals, ao]).astype(np.uint8).tobytes()).decode('ascii')

    somas = np.load(os.path.join(data_folder, 'somas_data.npy')) if os.path.exists(os.path.join(data_folder, 'somas_data.npy')) else np.empty((0, 3))
    info = {'name': meta.get('image_path', os.path.basename(os.path.normpath(data_folder))) + tag, 'shape': list(image.shape),
            'xy_res': xy * downsample, 'z_res': z_res, 'downsample': downsample, 'n_neurons': len(neurons),
            'somas': (somas / np.array([1, downsample, downsample])).round(1).tolist(), 'has_dapi': dapi is not None,
            'grid3d': [xy * f3, xy * f3, z_res * fz]}
    html = (_TEMPLATE.replace('__INFO__', json.dumps(info)).replace('__NEURONS__', json.dumps(neurons))
            .replace('__SLICES__', json.dumps(slices)).replace('__PROJECTION__', json.dumps(projection))
            .replace('__POINTS__', json.dumps(points3d)).replace('__SHADE__', json.dumps(shade3d)))
    with open(out_html, 'w') as f:
        f.write(html)
    print(f"Saved {out_html} ({os.path.getsize(out_html) / 1e6:.1f} MB, {len(neurons)} neurons, {image.shape[0]} slices, "
          f"{len(points)} 3D points on a {xy * f3:.2f} x {xy * f3:.2f} x {z_res * fz:.2f} um grid)")
    return out_html


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neuron Stack Viewer</title>
<style>
:root{--bg:#111316;--panel:#1b1e23;--line:#2c3038;--text:#e6e8eb;--muted:#9aa1ab;--accent:#7cc4ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:12px 16px;border-bottom:1px solid var(--line)}h1{font-size:16px;margin:0}header p{margin:2px 0 0;color:var(--muted);font-size:12px}
h2{font-size:13px;margin:0 0 6px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
main{padding:16px}.views{display:flex;gap:16px;flex-wrap:wrap}.panel{flex:1 1 420px;min-width:0}
.wrap{position:relative;width:100%;aspect-ratio:1;background:#000;border:1px solid var(--line)}
.wrap canvas{position:absolute;inset:0;width:100%;height:100%}#w2 canvas{image-rendering:pixelated}#c3{cursor:grab}
.controls{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;margin:0 0 8px;min-height:62px}
.controls label{display:flex;gap:6px;align-items:center;color:var(--muted)}input[type=range]{accent-color:var(--accent)}
#zrow{display:flex;gap:8px;align-items:center;width:100%}#z{flex:1}
#tip{position:absolute;pointer-events:none;background:#000c;padding:3px 6px;border-radius:4px;font-size:12px;display:none;white-space:nowrap}
.tablebox{margin-top:16px;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px;max-height:40vh;overflow:auto}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{text-align:right;padding:3px 6px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
tr{cursor:pointer}tr.sel{background:#2b3a4a}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
button{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:4px;padding:4px 10px;cursor:pointer}
.hint{color:var(--muted);font-size:12px;margin:6px 0 0}#nogl{position:absolute;inset:0;display:none;align-items:center;justify-content:center;color:var(--muted)}
</style></head><body>
<header><h1 id="title"></h1><p id="sub"></p></header>
<main><div class="views">
<section class="panel"><h2>Slices</h2>
<div class="controls"><div id="zrow"><button id="zm">&minus;</button><input id="z" type="range" min="0" value="0"><button id="zp">+</button><span id="zl"></span></div>
<label><input id="proj" type="checkbox"> max projection</label>
<label>overlay <input id="op" type="range" min="0" max="100" value="55"></label>
<label id="dapil"><input id="dapi" type="checkbox"> DAPI (blue)</label>
<label><input id="somas" type="checkbox" checked> soma centres</label>
<button id="clear">show all neurons</button></div>
<div class="wrap" id="w2"><canvas id="cimg"></canvas><canvas id="clab"></canvas><canvas id="cmark"></canvas><div id="tip"></div></div>
<p class="hint">Hover a neuron for its id; click it (or a table row) to show only that neuron, in 2D and 3D; arrow keys / mouse wheel change the slice.</p>
</section>
<section class="panel"><h2>3D morphology</h2>
<div class="controls"><label><input id="unas" type="checkbox"> unassigned segmentation (grey)</label>
<label>point size <input id="psz" type="range" min="3" max="40" value="12"></label>
<label>z stretch <input id="zst" type="range" min="5" max="30" value="10"></label>
<label><input id="shade" type="checkbox" checked> shading</label>
<button id="reset3">reset view</button></div>
<div class="wrap" id="w3"><canvas id="c3"></canvas><div id="nogl">WebGL is not available in this browser.</div></div>
<p class="hint">Drag to rotate, wheel to zoom, shift-drag to pan, double-click to reset. Starts seen from the top, like the slices.</p>
</section></div>
<div class="tablebox"><div id="count"></div><table><thead><tr><th>neuron</th><th>volume µm³</th><th>extent µm</th><th>z</th></tr></thead><tbody id="tb"></tbody></table></div></main>
<script>
const INFO=__INFO__, NEURONS=__NEURONS__, SLICES=__SLICES__, PROJ=__PROJECTION__, POINTS=__POINTS__, SHADE=__SHADE__;
const $=id=>document.getElementById(id), N=INFO.shape[1]/INFO.downsample|0;
const color=k=>{const h=(k*137.508)%360;return `hsl(${h},85%,58%)`};
const rgbOf=(()=>{const c=document.createElement('canvas').getContext('2d');const cache={};return k=>cache[k]||(c.fillStyle=color(k),c.fillRect(0,0,1,1),cache[k]=c.getImageData(0,0,1,1).data.slice(0,3))})();
let sel=null; const labCache=new Map(), imgCache=new Map();
$('title').textContent=`${INFO.name}: ${INFO.n_neurons} neurons`;
$('sub').textContent=`${INFO.shape[0]} slices × ${INFO.shape[1]}×${INFO.shape[2]} px (slices shown at ${N} px, ${INFO.xy_res.toFixed(3)} µm/px; z step ${INFO.z_res} µm; 3D grid ${INFO.grid3d[0].toFixed(2)} µm)`;
if(!INFO.has_dapi) $('dapil').style.display='none';
for(const c of ['cimg','clab','cmark']){$(c).width=N;$(c).height=N}
$('z').max=INFO.shape[0]-1; $('z').value=Math.floor(INFO.shape[0]/2);
if(location.hash==='#projection') $('proj').checked=true;
function loadImg(src){return new Promise(r=>{const i=new Image();i.onload=()=>r(i);i.src=src})}
async function labels(key,b64){if(labCache.has(key))return labCache.get(key);const i=await loadImg('data:image/png;base64,'+b64);
 const c=document.createElement('canvas');c.width=N;c.height=N;const x=c.getContext('2d');x.drawImage(i,0,0);const d=x.getImageData(0,0,N,N).data;
 const L=new Uint16Array(N*N);for(let p=0;p<N*N;p++)L[p]=d[4*p]*256+d[4*p+1];labCache.set(key,L);return L}
async function image(key,b64){if(imgCache.has(key))return imgCache.get(key);const i=await loadImg('data:image/jpeg;base64,'+b64);imgCache.set(key,i);return i}
let cur=null, sel3=undefined;
async function draw(){const proj=$('proj').checked, z=+$('z').value, s=proj?PROJ:SLICES[z], useD=$('dapi').checked&&INFO.has_dapi;
 $('zl').textContent=proj?'projection':`z ${z} / ${INFO.shape[0]-1}`; $('z').disabled=proj;
 const key=proj?'p':z, im=await image(key+(useD?'d':''),useD?s.dapi:s.img), L=await labels(key,s.lab); cur=L;
 $('cimg').getContext('2d').drawImage(im,0,0,N,N);
 const x=$('clab').getContext('2d'), out=x.createImageData(N,N), a=Math.round(+$('op').value*2.55);
 for(let p=0;p<N*N;p++){const k=L[p];if(!k||(sel!==null&&k!==sel))continue;const c=rgbOf(k);out.data[4*p]=c[0];out.data[4*p+1]=c[1];out.data[4*p+2]=c[2];out.data[4*p+3]=a}
 x.putImageData(out,0,0);
 const m=$('cmark').getContext('2d');m.clearRect(0,0,N,N);
 if($('somas').checked){m.strokeStyle='#fff';m.lineWidth=Math.max(1,N/700);for(const [sz,sy,sx] of INFO.somas){if(!proj&&Math.abs(sz-z)>3)continue;m.beginPath();m.arc(sx,sy,N/160,0,7);m.stroke()}}
 if(sel!==null){const n=NEURONS.find(n=>n.id===sel);if(n){m.strokeStyle='#fff';m.setLineDash([6,4]);m.beginPath();m.arc(n.centre[1],n.centre[0],n.extent_um/INFO.xy_res*0.6+10,0,7);m.stroke();m.setLineDash([])}}
 document.querySelectorAll('#tb tr').forEach(tr=>tr.classList.toggle('sel',+tr.dataset.id===sel));
 if(sel!==sel3){sel3=sel;update3d(true)}}
function pick(e){const r=$('w2').getBoundingClientRect(),px=Math.floor((e.clientX-r.left)/r.width*N),py=Math.floor((e.clientY-r.top)/r.height*N);
 return cur&&px>=0&&py>=0&&px<N&&py<N?cur[py*N+px]:0}
$('w2').addEventListener('mousemove',e=>{const k=pick(e),t=$('tip'),r=$('w2').getBoundingClientRect();
 if(k){const n=NEURONS.find(n=>n.id===k);t.textContent=`neuron ${k}: ${n?n.volume_um3:'?'} µm³`;t.style.left=(e.clientX-r.left+12)+'px';t.style.top=(e.clientY-r.top+12)+'px';t.style.display='block'}else t.style.display='none'});
$('w2').addEventListener('mouseleave',()=>$('tip').style.display='none');
$('w2').addEventListener('click',e=>{const k=pick(e);sel=k?(sel===k?null:k):null;draw()});
$('w2').addEventListener('wheel',e=>{if($('proj').checked)return;e.preventDefault();$('z').value=+$('z').value+(e.deltaY>0?1:-1);draw()},{passive:false});
document.addEventListener('keydown',e=>{if(e.key==='ArrowUp'||e.key==='ArrowRight'){$('z').value=+$('z').value+1;draw()}if(e.key==='ArrowDown'||e.key==='ArrowLeft'){$('z').value=+$('z').value-1;draw()}});
for(const id of ['z','proj','op','dapi','somas'])$(id).addEventListener('input',draw);
$('zm').onclick=()=>{$('z').value=+$('z').value-1;draw()};$('zp').onclick=()=>{$('z').value=+$('z').value+1;draw()};
$('clear').onclick=()=>{sel=null;draw()};
$('count').textContent=`${NEURONS.length} neurons (click a row to isolate it)`;
const tb=$('tb');for(const n of [...NEURONS].sort((a,b)=>b.volume_um3-a.volume_um3)){const tr=document.createElement('tr');tr.dataset.id=n.id;
 tr.innerHTML=`<td><span class="sw" style="background:${color(n.id)}"></span>${n.id}</td><td>${n.volume_um3}</td><td>${n.extent_um}</td><td>${n.z[0]}–${n.z[1]}</td>`;
 tr.onclick=()=>{sel=sel===n.id?null:n.id;if(sel!==null&&!$('proj').checked)$('z').value=Math.round((n.z[0]+n.z[1])/2);draw()};tb.appendChild(tr)}

// ---- 3D view: WebGL point cloud, one point per labelled voxel of the coarse grid ----
const c3=$('c3'), gl=c3.getContext('webgl',{antialias:true});
let update3d=()=>{};
if(!gl){$('nogl').style.display='flex'}else{
 const raw=atob(POINTS), bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
 const P=new Uint16Array(bytes.buffer), NP=P.length/4, G=INFO.grid3d;
 const lab=new Uint16Array(NP), pos=new Float32Array(NP*3), col=new Float32Array(NP*3), vis=new Float32Array(NP);
 const box={}; // label -> [minx,miny,minz,maxx,maxy,maxz] in um
 for(let i=0;i<NP;i++){const x=P[4*i]*G[0], y=-P[4*i+1]*G[1], z=-P[4*i+2]*G[2], k=P[4*i+3];lab[i]=k;pos[3*i]=x;pos[3*i+1]=y;pos[3*i+2]=z;
  const c=k?rgbOf(k):[140,140,140];col[3*i]=c[0]/255;col[3*i+1]=c[1]/255;col[3*i+2]=c[2]/255;
  const b=box[k]||(box[k]=[1e9,1e9,1e9,-1e9,-1e9,-1e9]);b[0]=Math.min(b[0],x);b[1]=Math.min(b[1],y);b[2]=Math.min(b[2],z);b[3]=Math.max(b[3],x);b[4]=Math.max(b[4],y);b[5]=Math.max(b[5],z)}
 const sh=(t,s)=>{const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);return o};
 const prog=gl.createProgram();
 gl.attachShader(prog,sh(gl.VERTEX_SHADER,`attribute vec3 p;attribute vec3 c;attribute float v;attribute vec4 a;uniform mat4 m;uniform mat3 r;uniform float ps,zs,dn,df,sh;varying vec3 vc;
  void main(){if(v<.5){gl_Position=vec4(2.,2.,2.,1.);gl_PointSize=0.;return;}gl_Position=m*vec4(p.x,p.y,p.z*zs,1.);
  gl_PointSize=max(1.5,ps/gl_Position.w);float d=clamp((gl_Position.w-dn)/(df-dn),0.,1.);
  if(sh<.5){vc=c*mix(1.15,.45,d);return;}
  vec3 n=a.xyz*2.-1.;n=normalize(r*vec3(n.x,n.y,n.z/zs));
  vec3 L=normalize(vec3(-.45,.55,.7)),H=normalize(L+vec3(0.,0.,1.));
  float df_=max(dot(n,L),0.)+.25*max(dot(n,-L),0.),sp=pow(max(dot(n,H),0.),28.);
  vc=(c*(.28+.85*df_)*mix(.45,1.,a.w)+.35*sp*a.w)*mix(1.,.55,d);}`));
 gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,`precision mediump float;varying vec3 vc;void main(){vec2 q=gl_PointCoord-.5;float e=dot(q,q);if(e>.25)discard;gl_FragColor=vec4(vc*(1.-.9*e),1.);}`));
 gl.linkProgram(prog);gl.useProgram(prog);
 const buf=(data,name,size)=>{const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);const a=gl.getAttribLocation(prog,name);gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,size,gl.FLOAT,false,0,0);return b};
 buf(pos,'p',3);buf(col,'c',3);const vbuf=buf(vis,'v',1);
 {const s=atob(SHADE),sb=new Uint8Array(s.length);for(let i=0;i<s.length;i++)sb[i]=s.charCodeAt(i);
  const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,sb,gl.STATIC_DRAW);
  const a=gl.getAttribLocation(prog,'a');gl.enableVertexAttribArray(a);gl.vertexAttribPointer(a,4,gl.UNSIGNED_BYTE,true,0,0)}
 const U=n=>gl.getUniformLocation(prog,n);
 const mul=(a,b)=>{const o=new Float32Array(16);for(let i=0;i<4;i++)for(let j=0;j<4;j++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+j]*b[i*4+k];o[i*4+j]=s}return o};
 const T=(x,y,z)=>new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,x,y,z,1]);
 const RX=t=>{const c=Math.cos(t),s=Math.sin(t);return new Float32Array([1,0,0,0,0,c,s,0,0,-s,c,0,0,0,0,1])};
 const RY=t=>{const c=Math.cos(t),s=Math.sin(t);return new Float32Array([c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1])};
 const persp=(f,a,n,fa)=>{const t=1/Math.tan(f/2);return new Float32Array([t/a,0,0,0,0,t,0,0,0,0,(fa+n)/(n-fa),-1,0,0,2*fa*n/(n-fa),0])};
 const FOV=0.7; let yaw=0,pitch=0,dist=1,pan=[0,0],centre=[0,0,0],radius=1,needs=true;
 function frame(k){const ks=k===null?Object.keys(box).filter(x=>+x>0):[k];let b=[1e9,1e9,1e9,-1e9,-1e9,-1e9];
  for(const q of ks){const c=box[q];if(!c)continue;for(let i=0;i<3;i++){b[i]=Math.min(b[i],c[i]);b[i+3]=Math.max(b[i+3],c[i+3])}}
  const zs=+$('zst').value/10;centre=[(b[0]+b[3])/2,(b[1]+b[4])/2,(b[2]+b[5])/2*zs];
  radius=Math.max(5,0.5*Math.hypot(b[3]-b[0],b[4]-b[1],(b[5]-b[2])*zs));dist=radius/Math.tan(FOV/2)*1.05;pan=[0,0]}
 update3d=(refit)=>{const un=$('unas').checked;for(let i=0;i<NP;i++){const k=lab[i];vis[i]=sel===null?(k?1:(un?1:0)):(k===sel?1:0)}
  gl.bindBuffer(gl.ARRAY_BUFFER,vbuf);gl.bufferSubData(gl.ARRAY_BUFFER,0,vis);if(refit)frame(sel);needs=true};
 function render(){if(!needs){requestAnimationFrame(render);return}needs=false;
  const dpr=window.devicePixelRatio||1,w=c3.clientWidth*dpr|0,h=c3.clientHeight*dpr|0;if(c3.width!==w||c3.height!==h){c3.width=w;c3.height=h}
  gl.viewport(0,0,w,h);gl.clearColor(0,0,0,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);
  const zs=+$('zst').value/10, near=Math.max(0.1,dist-radius*3), far=dist+radius*3;
  const view=mul(T(pan[0],pan[1],-dist),mul(RX(pitch),mul(RY(yaw),T(-centre[0],-centre[1],-centre[2]))));
  gl.uniformMatrix4fv(U('m'),false,mul(persp(FOV,w/h,near,far),view));
  const R=mul(RX(pitch),RY(yaw));gl.uniformMatrix3fv(U('r'),false,new Float32Array([R[0],R[1],R[2],R[4],R[5],R[6],R[8],R[9],R[10]]));
  gl.uniform1f(U('sh'),$('shade').checked?1:0);
  gl.uniform1f(U('ps'),+$('psz').value/10*Math.max(G[0],G[2]*zs)*h/(2*Math.tan(FOV/2)));gl.uniform1f(U('zs'),zs);gl.uniform1f(U('dn'),dist-radius);gl.uniform1f(U('df'),dist+radius);
  gl.drawArrays(gl.POINTS,0,NP);requestAnimationFrame(render)}
 let drag=null;
 c3.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,shift:e.shiftKey};c3.style.cursor='grabbing'});
 window.addEventListener('mouseup',()=>{drag=null;c3.style.cursor='grab'});
 window.addEventListener('mousemove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.x=e.clientX;drag.y=e.clientY;
  if(drag.shift){const s=2*dist*Math.tan(FOV/2)/c3.clientHeight;pan[0]+=dx*s;pan[1]-=dy*s}else{yaw+=dx*0.01;pitch=Math.max(-3.1,Math.min(3.1,pitch+dy*0.01))}needs=true});
 c3.addEventListener('wheel',e=>{e.preventDefault();dist*=e.deltaY>0?1.1:1/1.1;needs=true},{passive:false});
 const reset=()=>{yaw=0;pitch=0;frame(sel);needs=true};c3.addEventListener('dblclick',reset);$('reset3').onclick=reset;
 $('unas').addEventListener('input',()=>update3d(false));$('psz').addEventListener('input',()=>needs=true);$('shade').addEventListener('input',()=>needs=true);$('zst').addEventListener('input',()=>{frame(sel);needs=true});
 window.addEventListener('resize',()=>needs=true);
 requestAnimationFrame(render)}
draw();
</script></body></html>
"""


def main():
    parser = argparse.ArgumentParser(description="Write an HTML viewer of the separated neurons of a processed image.")
    parser.add_argument("data_folder", help="Folder with processed_image.npy, processed_metadata.npy and neuron_labels.tiff")
    parser.add_argument("--dapi-tif", default=None, help="Multi-channel TIFF to take the DAPI channel (index 0) from, optional")
    parser.add_argument("-o", "--out", default=None, help="Output .html (default: <data_folder>/neuron_viewer<tag>.html)")
    parser.add_argument("--tag", default='', help="segment_neurons file tag, e.g. _sigma2 (default: none)")
    args = parser.parse_args()
    dapi = None
    if args.dapi_tif:
        with tiff.TiffFile(args.dapi_tif) as t:
            series = t.series[0]
            dapi = np.take(series.asarray(), 0, axis=series.axes.index('C'))
    make_viewer(args.data_folder, args.out, dapi=dapi, tag=args.tag)


if __name__ == "__main__":
    main()
