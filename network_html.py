"""HTML page comparing the activity of the morphology and soma-only networks (called by bsb_network.py).

One self-contained page (data inline; Plotly from cdnjs): replay of the NEST spikes in two synced 3D views (somas light up when they spike,
neurite skeletons in the morphology network, connections as lines), raster plots, population rate with detected network bursts,
distributions (firing rate, in-degree, connection distance) and a table of activity statistics (mean +- SD over repeats).
"""
import os
import json
import urllib.request
import numpy as np

PLOTLY_URL = 'https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js'
PLOTLY_CACHE = os.path.join('results', 'network', 'plotly-2.35.2.min.js')


def _plotly_tag():
    """Plotly inlined (page works offline, e.g. in a presentation); downloaded once to PLOTLY_CACHE, CDN <script> as fallback."""
    if not os.path.exists(PLOTLY_CACHE):
        try:
            os.makedirs(os.path.dirname(PLOTLY_CACHE), exist_ok=True)
            with urllib.request.urlopen(PLOTLY_URL, timeout=30) as r:
                js = r.read()
            if b'plotly' not in js[:2000].lower():
                raise ValueError('unexpected content')
            with open(PLOTLY_CACHE, 'wb') as f:
                f.write(js)
        except Exception as e:
            print(f"Plotly not inlined ({e}); the page loads it from {PLOTLY_URL}")
            return f'<script src="{PLOTLY_URL}"></script>'
    return '<script>' + open(PLOTLY_CACHE, encoding='utf-8').read().replace('</script', '<\\/script') + '</script>'

PALETTE = {'morph': '#e8a33d', 'soma': '#4fa3e0'}


def _skeleton_lines(swcs, max_points=120_000):
    """x, y, z lists with None between segments (parent -> child), plus the neuron index per point (for colour)."""
    xs, ys, zs, cs = [], [], [], []
    total = sum(len(a) for a in swcs)
    step = max(1, int(np.ceil(3 * total / max_points)))   # keep every step-th node along each path if very large
    for k, a in enumerate(swcs):
        par = a[:, 6].astype(int) - 1
        xyz = a[:, 2:5]
        keep = np.zeros(len(a), bool); keep[::step] = True; keep[0] = True
        nchild = np.bincount(par[1:], minlength=len(a)); keep |= (nchild != 1)
        for i in range(1, len(a)):
            if not keep[i]:
                continue
            p = par[i]
            while p > 0 and not keep[p]:
                p = par[p]
            for q in (p, i):
                xs.append(round(float(xyz[q, 0]), 2)); ys.append(round(float(xyz[q, 1]), 2)); zs.append(round(float(xyz[q, 2]), 2)); cs.append(k)
            xs.append(None); ys.append(None); zs.append(None); cs.append(k)
    return xs, ys, zs, cs


def _pair_lines(pos, pairs):
    """Unordered pairs as line segments; reciprocal pairs drawn once."""
    u = {tuple(sorted(p)) for p in pairs.tolist()}
    xs, ys, zs = [], [], []
    for i, j in sorted(u):
        for q in (i, j):
            xs.append(round(float(pos[q, 0]), 2)); ys.append(round(float(pos[q, 1]), 2)); zs.append(round(float(pos[q, 2]), 2))
        xs.append(None); ys.append(None); zs.append(None)
    return {'x': xs, 'y': ys, 'z': zs}


def _conn_stats(pos, pairs, n):
    C = np.zeros((n, n), bool); C[pairs[:, 0], pairs[:, 1]] = True
    d = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1) if len(pairs) else np.zeros(0)
    U = C | C.T
    deg = U.sum(1)
    tri = np.einsum('ij,jk,ki->i', U.astype(int), U.astype(int), U.astype(int))
    clust = np.where(deg > 1, tri / np.maximum(deg * (deg - 1), 1), 0.0)
    return {'connections': int(C.sum()), 'reciprocal_fraction': float((C & C.T).sum() / max(C.sum(), 1)),
            'mean_indegree': float(C.sum(0).mean()), 'isolated_cells': int((deg == 0).sum()),
            'mean_distance_um': float(d.mean()) if len(d) else 0.0, 'clustering': float(clust.mean()),
            'indegree': C.sum(0).tolist(), 'distances': np.round(d, 1).tolist()}


def _finite(v):
    """Undefined statistics (NaN / inf, e.g. inter-burst CV with < 2 intervals) -> null in the page's JSON."""
    if isinstance(v, dict):
        return {k: _finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_finite(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    return v


def write_html(path, summary, pos, swcs, conn_morph, conn_soma, spikes, duration):
    from bsb_network import detect_bursts
    n = len(pos)
    order = np.argsort(pos[:, 0]).tolist()          # raster rows sorted by x (spatial layout)
    rank = np.empty(n, int); rank[order] = np.arange(n)
    nets = {}
    for key, pairs in (('morph', conn_morph), ('soma', conn_soma)):
        t, s = spikes[key]
        per_cell = [np.round(np.sort(t[s == i]), 1).tolist() for i in range(n)]
        nets[key] = {'spikes': per_cell, 'raster_t': np.round(t, 1).tolist(), 'raster_y': rank[s].tolist(),
                     'bursts': detect_bursts(t, s, n, duration), 'rates': (np.bincount(s, minlength=n) / (duration / 1000)).round(2).tolist(),
                     'lines': _pair_lines(pos, pairs), 'conn': _conn_stats(pos, pairs, n)}
    sx, sy, sz, sc = _skeleton_lines(swcs)
    data = {'n': n, 'duration': duration, 'pos': np.round(pos, 2).tolist(), 'nets': nets, 'summary': summary,
            'skeleton': {'x': sx, 'y': sy, 'z': sz, 'c': sc}, 'palette': PALETTE}
    html = HTML.replace('__DATA__', json.dumps(_finite(data), separators=(',', ':'), allow_nan=False))
    html = html.replace('__TITLE__', f"Network activity · {summary['image']}").replace('__PLOTLY__', _plotly_tag())
    with open(path, 'w') as f:
        f.write(html)
    print(f"Wrote {path}")


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
__PLOTLY__
<style>
:root{--bg:#101215;--panel:#181b20;--line:#2a2f37;--text:#e6e8eb;--muted:#9aa1aa;--morph:#e8a33d;--soma:#4fa3e0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:14px 18px 6px}h1{font-size:18px;margin:0 0 4px}h2{font-size:14px;margin:0 0 8px;font-weight:600}
.sub{color:var(--muted);max-width:1100px}
.wrap{padding:8px 18px 24px;display:flex;flex-direction:column;gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
.row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.row3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
@media (max-width:900px){.row,.row3{grid-template-columns:1fr}}
.tag{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:8px}
.controls button{background:#2b313a;color:var(--text);border:1px solid #3b4250;border-radius:5px;padding:5px 12px;cursor:pointer;font:inherit}
.controls input[type=range]{flex:1;min-width:200px}
#tval{font-variant-numeric:tabular-nums;min-width:90px}
.plot3d{height:520px}.plot{height:300px}.small{height:240px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500}
.note{color:var(--muted);font-size:12px}
.better{color:#fff;font-weight:600}
</style></head><body>
<header><h1>__TITLE__</h1><div class="sub" id="sub"></div></header>
<div class="wrap">
 <div class="card"><h2>Network activity replay</h2>
  <div class="controls">
   <button id="play">▶ play</button>
   <label>speed <select id="speed"><option value="0.1">0.1×</option><option value="0.25" selected>0.25×</option><option value="0.5">0.5×</option><option value="1">1× (real time)</option></select></label>
   <input type="range" id="time" min="0" step="1" value="0"><span id="tval"></span>
   <label><input type="checkbox" id="showconn" checked> connections</label>
   <label><input type="checkbox" id="showskel" checked> neurites</label>
  </div>
  <div class="row">
   <div><div><span class="tag" style="background:var(--morph)"></span><b>Morphology network</b> <span class="note" id="m_desc"></span></div><div id="g_morph" class="plot3d"></div></div>
   <div><div><span class="tag" style="background:var(--soma)"></span><b>Soma-only network</b> <span class="note" id="s_desc"></span></div><div id="g_soma" class="plot3d"></div></div>
  </div>
  <p class="note">Somas light up when they spike (decay 30 ms). Rotating one view rotates both. Spikes of run 0 (same Poisson input seed in both networks).</p>
 </div>
 <div class="card"><h2>Spikes</h2><div id="raster" class="plot" style="height:430px"></div>
  <div id="poprate" class="plot" style="height:220px"></div>
  <p class="note">Rows sorted by soma x position. Shaded: detected network bursts (≥ 20% of the cells spiking in 10 ms bins).</p></div>
 <div class="card"><h2>Comparison</h2><div id="table"></div>
  <div class="row3" style="margin-top:10px"><div id="h_rate" class="small"></div><div id="h_indeg" class="small"></div><div id="h_dist" class="small"></div></div></div>
</div>
<script>
const D=__DATA__;const P=D.palette;const N=D.n,T=D.duration;const S=D.summary;
const dark={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:'#d8dbe0',size:11}};
const ax={gridcolor:'#2a2f37',zerolinecolor:'#2a2f37'};
document.getElementById('sub').innerHTML=`${N} reconstructed neurons of <b>${S.image}</b> (${S.tag}), ${S.extent_um.map(v=>v.toFixed(0)).join(' × ')} µm. `+
 `Both networks: same cells, NEST aeif_cond_alpha (adaptation a ${S.params.adapt_a} nS, b ${S.params.adapt_b} pA, τw ${S.params.tau_w} ms), `+
 `excitatory synapses ${S.params.w_rec} nS / ${S.params.delay} ms, independent Poisson input ${S.params.input_rate} Hz × ${S.params.w_in} nS, ${T/1000} s. `+
 `<b>Morphology</b>: BSB VoxelIntersection of the reconstructed skeletons. <b>Soma-only</b>: p(d) = ${S.fit.p0.toFixed(2)}·exp(−d²/2·${S.fit.sigma_um.toFixed(1)}²) `+
 `fitted to the morphology network (same expected number of connections and distance dependence, reciprocal).`;
const C={morph:D.nets.morph.conn,soma:D.nets.soma.conn};
document.getElementById('m_desc').textContent=`${C.morph.connections} connections, ${(100*C.morph.reciprocal_fraction).toFixed(0)}% reciprocal`;
document.getElementById('s_desc').textContent=`${C.soma.connections} connections, ${(100*C.soma.reciprocal_fraction).toFixed(0)}% reciprocal`;
// ---------- 3D
const X=D.pos.map(p=>p[0]),Y=D.pos.map(p=>p[1]),Z=D.pos.map(p=>p[2]);
const scene={xaxis:{title:'x (µm)',...ax,backgroundcolor:'#101215',color:'#9aa1aa'},yaxis:{title:'y (µm)',...ax,color:'#9aa1aa',autorange:'reversed'},
 zaxis:{title:'z (µm)',...ax,color:'#9aa1aa'},aspectmode:'data',camera:{eye:{x:0.2,y:-1.2,z:1.4}}};
const somaScale=[[0,'#38404d'],[0.25,'#6b5a3a'],[0.6,'#f0a020'],[1,'#fff6c8']];
function soma3d(){return {type:'scatter3d',mode:'markers',x:X,y:Y,z:Z,hovertext:X.map((_,i)=>'cell '+i),hoverinfo:'text',
 marker:{size:Array(N).fill(4),color:Array(N).fill(0),cmin:0,cmax:1,colorscale:somaScale,line:{width:0}},showlegend:false}}
function conn3d(k){const l=D.nets[k].lines;return {type:'scatter3d',mode:'lines',x:l.x,y:l.y,z:l.z,hoverinfo:'skip',
 line:{color:P[k],width:1.5},opacity:0.35,showlegend:false}}
const sk=D.skeleton;
const skel={type:'scatter3d',mode:'lines',x:sk.x,y:sk.y,z:sk.z,hoverinfo:'skip',line:{color:sk.c,colorscale:'Viridis',width:2},opacity:0.55,showlegend:false};
const lay3={...dark,margin:{l:0,r:0,t:0,b:0},scene};
Plotly.newPlot('g_morph',[skel,conn3d('morph'),soma3d()],lay3,{displaylogo:false,responsive:true});
Plotly.newPlot('g_soma',[conn3d('soma'),soma3d()],JSON.parse(JSON.stringify(lay3)),{displaylogo:false,responsive:true});
const SOMA_IDX={g_morph:2,g_soma:1};
let syncing=false;
function syncCam(src,dst){document.getElementById(src).on('plotly_relayout',e=>{if(syncing||!e['scene.camera'])return;syncing=true;
 Plotly.relayout(dst,{'scene.camera':e['scene.camera']}).then(()=>syncing=false)})}
syncCam('g_morph','g_soma');syncCam('g_soma','g_morph');
document.getElementById('showconn').onchange=e=>{Plotly.restyle('g_morph',{visible:e.target.checked},[1]);Plotly.restyle('g_soma',{visible:e.target.checked},[0])};
document.getElementById('showskel').onchange=e=>Plotly.restyle('g_morph',{visible:e.target.checked},[0]);
// ---------- raster + population rate
const order=[...Array(N).keys()];
function rasterTrace(k,yaxis){const n=D.nets[k];return {type:'scattergl',mode:'markers',x:n.raster_t,y:n.raster_y,yaxis,xaxis:'x',
 marker:{size:2.2,color:P[k]},hoverinfo:'skip',name:k==='morph'?'morphology':'soma-only'}}
function burstShapes(k,yref){return D.nets[k].bursts.map(b=>({type:'rect',xref:'x',yref,x0:b[0],x1:b[1],y0:0,y1:1,fillcolor:P[k],opacity:0.12,line:{width:0},layer:'below'}))}
const cursor=t=>({type:'line',xref:'x',yref:'paper',x0:t,x1:t,y0:0,y1:1,line:{color:'#fff',width:1}});
const rlay={...dark,margin:{l:55,r:10,t:10,b:35},showlegend:false,xaxis:{...ax,title:'time (ms)',range:[0,T],anchor:'y2'},
 yaxis:{...ax,domain:[0.53,1],title:'morphology<br>cell (by x)',range:[-1,N]},yaxis2:{...ax,domain:[0,0.47],title:'soma-only<br>cell (by x)',range:[-1,N]},
 shapes:[...burstShapes('morph','y domain'),...burstShapes('soma','y2 domain'),cursor(0)]};
Plotly.newPlot('raster',[rasterTrace('morph','y'),rasterTrace('soma','y2')],rlay,{displaylogo:false,responsive:true});
const BIN=5;const edges=[...Array(Math.ceil(T/BIN)).keys()].map(i=>i*BIN);
function popRate(k){const c=new Float64Array(edges.length);for(const t of D.nets[k].raster_t){const b=Math.min(edges.length-1,Math.floor(t/BIN));c[b]++}
 return Array.from(c,v=>v/N/(BIN/1000))}
const play2={...dark,margin:{l:55,r:10,t:10,b:35},xaxis:{...ax,title:'time (ms)',range:[0,T]},yaxis:{...ax,title:'rate (Hz / cell)'},legend:{orientation:'h',y:1.15},shapes:[cursor(0)]};
Plotly.newPlot('poprate',['morph','soma'].map(k=>({type:'scattergl',mode:'lines',x:edges,y:popRate(k),line:{color:P[k],width:1.2},name:k==='morph'?'morphology':'soma-only'})),play2,{displaylogo:false,responsive:true});
// link zoom of raster and rate
let linking=false;
function linkX(a,b){document.getElementById(a).on('plotly_relayout',e=>{if(linking)return;const r=e['xaxis.range[0]']!==undefined?[e['xaxis.range[0]'],e['xaxis.range[1]']]:(e['xaxis.autorange']?[0,T]:null);
 if(!r)return;linking=true;Plotly.relayout(b,{'xaxis.range':r}).then(()=>linking=false)})}
linkX('raster','poprate');linkX('poprate','raster');
// ---------- replay
const tin=document.getElementById('time');tin.max=T;const tval=document.getElementById('tval');
let t=0,playing=false,last=null;const ptr={morph:new Int32Array(N),soma:new Int32Array(N)};
function seek(nt){t=nt;for(const k of ['morph','soma']){const sp=D.nets[k].spikes;for(let i=0;i<N;i++){let lo=0,hi=sp[i].length;while(lo<hi){const m=(lo+hi)>>1;if(sp[i][m]<=t)lo=m+1;else hi=m}ptr[k][i]=lo}}render()}
let lastCursor=0;
function render(){tval.textContent=(t/1000).toFixed(3)+' s';tin.value=t;
 for(const [k,div] of [['morph','g_morph'],['soma','g_soma']]){const sp=D.nets[k].spikes;const col=new Array(N),sz=new Array(N);
  for(let i=0;i<N;i++){while(ptr[k][i]<sp[i].length&&sp[i][ptr[k][i]]<=t)ptr[k][i]++;const j=ptr[k][i]-1;const a=j>=0?Math.exp(-(t-sp[i][j])/30):0;col[i]=a;sz[i]=4+9*a}
  Plotly.restyle(div,{'marker.color':[col],'marker.size':[sz]},[SOMA_IDX[div]])}
 const now=performance.now();if(now-lastCursor>120){lastCursor=now;
  Plotly.relayout('raster',{'shapes[shapes.length]':null,shapes:[...rlay.shapes.slice(0,-1),cursor(t)]});Plotly.relayout('poprate',{shapes:[cursor(t)]})}}
function frame(ts){if(!playing)return;if(last!==null){t+=(ts-last)*parseFloat(document.getElementById('speed').value);if(t>=T){t=0;seek(0)}}last=ts;render();requestAnimationFrame(frame)}
document.getElementById('play').onclick=()=>{playing=!playing;document.getElementById('play').textContent=playing?'❚❚ pause':'▶ play';last=null;if(playing)requestAnimationFrame(frame)};
tin.oninput=()=>seek(+tin.value);
document.getElementById('raster').on('plotly_click',e=>{if(e.points&&e.points[0])seek(e.points[0].x)});
seek(0);
// ---------- comparison table
const M=[['mean_rate_hz','mean firing rate (Hz)',2],['cv_isi_median','ISI CV (median cell)',2],['synchrony_chi','synchrony χ (5 ms)',3],
 ['fano_pop_5ms','population Fano factor (5 ms)',1],['bursts_per_s','network bursts / s',2],['burst_duration_ms','burst duration (ms)',0],
 ['burst_participation','cells per burst (fraction)',2],['spikes_in_bursts','spikes inside bursts (fraction)',2],['ibi_cv','inter-burst interval CV',2]];
const R=S.params.repeats;
let h=`<table><tr><th></th><th><span class="tag" style="background:var(--morph)"></span>morphology</th><th><span class="tag" style="background:var(--soma)"></span>soma-only</th></tr>`;
const fmt=(v,d)=>v===null||v===undefined||Number.isNaN(v)?'–':v.toFixed(d);
h+=`<tr><td colspan="3" class="note">Connectivity</td></tr>`;
for(const [k,l,d] of [['connections','connections',0],['mean_indegree','mean in-degree',2],['reciprocal_fraction','reciprocal fraction',2],['isolated_cells','unconnected cells',0],['mean_distance_um','mean connection distance (µm)',1],['clustering','clustering coefficient',3]])
 h+=`<tr><td>${l}</td><td>${fmt(C.morph[k],d)}</td><td>${fmt(C.soma[k],d)}</td></tr>`;
h+=`<tr><td colspan="3" class="note">Activity: mean ± SD over ${R} runs (input seeds; the soma-only network is also re-drawn each run)</td></tr>`;
for(const [k,l,d] of M){h+=`<tr><td>${l}</td>`;for(const n of ['morph','soma']){const m=S[n].mean[k],s=S[n].sd[k];h+=`<td>${fmt(m,d)}${R>1?' ± '+fmt(s,d):''}</td>`}h+='</tr>'}
document.getElementById('table').innerHTML=h+'</table>';
function hist(div,key,title,xt,src){const tr=['morph','soma'].map(k=>({type:'histogram',x:src(k),name:k==='morph'?'morphology':'soma-only',opacity:0.6,marker:{color:P[k]}}));
 Plotly.newPlot(div,tr,{...dark,barmode:'overlay',margin:{l:45,r:10,t:30,b:40},title:{text:title,font:{size:12}},xaxis:{...ax,title:xt},yaxis:{...ax,title:'count'},legend:{orientation:'h',x:1,xanchor:'right',y:1,bgcolor:'rgba(0,0,0,0)'}},{displaylogo:false,responsive:true})}
hist('h_rate','','Firing rate per cell (run 0)','Hz',k=>D.nets[k].rates);
hist('h_indeg','','In-degree','connections',k=>C[k].indegree);
hist('h_dist','','Connection distance (soma to soma)','µm',k=>C[k].distances);
</script></body></html>
"""


def rebuild_html(out_dir, data_folder):
    """Write network_activity.html again from the outputs saved by bsb_network.py (no new simulation)."""
    path = os.path.join(out_dir, 'network_activity.html')
    if os.path.exists(path):
        raise FileExistsError(f"Not overwriting {path}")
    summary = json.load(open(os.path.join(out_dir, 'summary.json')))
    net, sp = np.load(os.path.join(out_dir, 'network.npz')), np.load(os.path.join(out_dir, 'spikes.npz'))
    from bsb_network import load_swcs
    swcs = load_swcs(data_folder, summary['tag'])
    spikes = {k: (sp[f'{k}_t'], sp[f'{k}_s']) for k in ('morph', 'soma')}
    write_html(path, summary, net['positions'], [swcs[k] for k in net['swc_index']], net['conn_morph'], net['conn_soma'],
               spikes, summary['params']['duration'])


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Rebuild network_activity.html from a bsb_network.py output folder.")
    ap.add_argument('out_dir', help='results/network/<stem><tag>/')
    ap.add_argument('data_folder', help='data/<stem>/ (SWC files for the neurite display)')
    a = ap.parse_args()
    rebuild_html(a.out_dir, a.data_folder)
