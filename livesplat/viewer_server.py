"""Browser-based operator viewer for the live splat.

Runs a small HTTP server (Python stdlib only -- no extra pip deps, so it starts
on the SplaTAM box with zero setup) that an operator opens from a laptop over
the network. The page:

* streams the live splat as a coloured point cloud (binary blob, re-fetched
  whenever the SLAM loop publishes a new snapshot),
* shows live status (frames, Gaussians, ROI coverage, next-best-view hints),
* lets the operator position an axis-aligned box over an object and press
  "Refine" to request a higher-quality splat of that region.

Transport choices are deliberately dumb-simple for robustness over a field
link: JSON polling for status, a versioned binary endpoint for geometry, and a
single JSON POST for the ROI. No websockets, no build step.

The 3D front-end loads three.js from a CDN, which is fine for a real operator
laptop with internet. If the operator laptop is fully offline, vendor three.js
next to this file and set ``THREE_JS_URL`` to a local path.
"""

from __future__ import annotations

import io
import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import numpy as np

from livesplat.bridge import LiveSplatBridge
from livesplat.roi import RefineRequest


THREE_JS_URL = "https://unpkg.com/three@0.160.0/build/three.module.js"
ORBIT_URL = "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js"


def _pack_snapshot(snapshot: dict) -> bytes:
    """Binary layout: [uint32 n][n*3 float32 xyz][n*3 uint8 rgb]."""
    xyz = np.ascontiguousarray(snapshot["xyz"], dtype="<f4")
    rgb = np.ascontiguousarray(snapshot["rgb"], dtype="u1")
    n = xyz.shape[0]
    buf = io.BytesIO()
    buf.write(struct.pack("<I", n))
    buf.write(xyz.tobytes())
    buf.write(rgb.tobytes())
    return buf.getvalue()


class _Handler(BaseHTTPRequestHandler):
    bridge: LiveSplatBridge = None  # injected by OperatorViewerServer

    # Silence the default per-request stderr logging (noisy under polling).
    def log_message(self, *args, **kwargs):
        return

    def _send(self, code, content_type, body: bytes, extra_headers=None):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, "text/html; charset=utf-8", _PAGE.encode("utf-8"))
        elif self.path.startswith("/status"):
            body = json.dumps(self.bridge.get_status()).encode("utf-8")
            self._send(200, "application/json", body)
        elif self.path.startswith("/snapshot_version"):
            v = self.bridge.snapshot_version()
            self._send(200, "application/json", json.dumps({"version": v}).encode())
        elif self.path.startswith("/snapshot.bin"):
            snap = self.bridge.get_snapshot()
            if snap is None:
                self._send(200, "application/octet-stream", struct.pack("<I", 0))
            else:
                self._send(200, "application/octet-stream", _pack_snapshot(snap))
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if not self.path.startswith("/roi"):
            self._send(404, "text/plain", b"not found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            req = RefineRequest.from_dict(payload)
            rid = self.bridge.submit_refine_request(req)
            self._send(200, "application/json", json.dumps({"request_id": rid}).encode())
        except Exception as e:  # noqa: BLE001 -- report to the operator, keep serving
            self._send(400, "application/json", json.dumps({"error": str(e)}).encode())


class OperatorViewerServer:
    """Owns the HTTP server thread and the shared bridge."""

    def __init__(self, bridge: LiveSplatBridge, host: str = "0.0.0.0", port: int = 8080):
        self.bridge = bridge
        self.host = host
        self.port = port
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        handler = type("BoundHandler", (_Handler,), {"bridge": self.bridge})
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()


# The operator page. Kept inline so the server is a single importable module.
_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Live Splat Operator</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font-family: system-ui, sans-serif; background:#0b0d10; color:#e6e6e6; }
  #app { position:fixed; inset:0; }
  #panel { position:fixed; top:12px; left:12px; width:300px; background:rgba(20,24,30,.92);
           border:1px solid #2a3038; border-radius:12px; padding:14px; font-size:13px; }
  h1 { font-size:15px; margin:0 0 8px; }
  .row { display:flex; justify-content:space-between; margin:3px 0; }
  .k { color:#8b98a8; } .v { color:#e6e6e6; font-variant-numeric:tabular-nums; }
  label { display:block; margin:8px 0 2px; color:#8b98a8; }
  input[type=range] { width:100%; }
  button { width:100%; margin-top:10px; padding:9px; border:0; border-radius:8px;
           background:#3b82f6; color:#fff; font-weight:600; cursor:pointer; }
  button.secondary { background:#374151; }
  #hints { margin-top:8px; font-size:12px; color:#a7f3d0; white-space:pre-line; }
  .bar { height:6px; background:#1f2530; border-radius:4px; overflow:hidden; margin-top:2px; }
  .bar > i { display:block; height:100%; background:#22c55e; width:0%; }
</style>
</head>
<body>
<div id="app"></div>
<div id="panel">
  <h1>Live Splat Operator</h1>
  <div class="row"><span class="k">Frame</span><span class="v" id="frame">-</span></div>
  <div class="row"><span class="k">Gaussians</span><span class="v" id="gauss">-</span></div>
  <div class="row"><span class="k">Preview pts</span><span class="v" id="pts">-</span></div>
  <div class="row"><span class="k">ROI coverage</span><span class="v" id="cov">-</span></div>
  <div class="bar"><i id="covbar"></i></div>

  <label>ROI center X <span id="cxv"></span></label><input type="range" id="cx" min="-5" max="5" step="0.02" value="0"/>
  <label>ROI center Y <span id="cyv"></span></label><input type="range" id="cy" min="-5" max="5" step="0.02" value="0"/>
  <label>ROI center Z <span id="czv"></span></label><input type="range" id="cz" min="-5" max="5" step="0.02" value="1"/>
  <label>ROI size <span id="szv"></span></label><input type="range" id="sz" min="0.1" max="4" step="0.02" value="0.6"/>
  <label>Quality (compute) <span id="qv"></span></label><input type="range" id="q" min="1" max="5" step="0.5" value="2"/>

  <button id="refine">Refine this region</button>
  <button id="reset" class="secondary">Reset view</button>
  <div id="hints"></div>
</div>

<script type="module">
import * as THREE from '""" + THREE_JS_URL + """';
import { OrbitControls } from '""" + ORBIT_URL + """';

const app = document.getElementById('app');
const renderer = new THREE.WebGLRenderer({ antialias:true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d10);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.01, 1000);
camera.position.set(2, 1.5, 3);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.AxesHelper(0.5));

let cloud = null;
const boxGeo = new THREE.BoxGeometry(1,1,1);
const box = new THREE.LineSegments(new THREE.EdgesGeometry(boxGeo),
             new THREE.LineBasicMaterial({ color:0x22c55e }));
scene.add(box);

function el(id){ return document.getElementById(id); }
function updateBox(){
  const cx=+el('cx').value, cy=+el('cy').value, cz=+el('cz').value, s=+el('sz').value;
  box.position.set(cx,cy,cz); box.scale.set(s,s,s);
  el('cxv').textContent=cx.toFixed(2); el('cyv').textContent=cy.toFixed(2);
  el('czv').textContent=cz.toFixed(2); el('szv').textContent=s.toFixed(2);
  el('qv').textContent=(+el('q').value).toFixed(1);
}
['cx','cy','cz','sz','q'].forEach(id=> el(id).addEventListener('input', updateBox));
updateBox();

el('reset').onclick=()=>{ camera.position.set(2,1.5,3); controls.target.set(0,0,1); };
el('refine').onclick=async()=>{
  const cx=+el('cx').value, cy=+el('cy').value, cz=+el('cz').value, s=+el('sz').value/2;
  const body={ roi:{ min_xyz:[cx-s,cy-s,cz-s], max_xyz:[cx+s,cy+s,cz+s], label:'operator' },
               quality:+el('q').value, guide_recapture:true };
  const r=await fetch('/roi',{method:'POST',headers:{'Content-Type':'application/json'},
                             body:JSON.stringify(body)});
  const j=await r.json();
  el('hints').textContent = j.request_id ? ('Refine request #'+j.request_id+' queued.') : ('Error: '+(j.error||'?'));
};

let lastVersion=-1;
async function pollSnapshot(){
  try{
    const vr=await fetch('/snapshot_version'); const vj=await vr.json();
    if(vj.version!==lastVersion){
      lastVersion=vj.version;
      const buf=await (await fetch('/snapshot.bin')).arrayBuffer();
      const dv=new DataView(buf); const n=dv.getUint32(0,true);
      if(n>0){
        const xyz=new Float32Array(buf, 4, n*3);
        const rgb=new Uint8Array(buf, 4+n*12, n*3);
        const cols=new Float32Array(n*3);
        for(let i=0;i<n*3;i++) cols[i]=rgb[i]/255;
        const g=new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(xyz,3));
        g.setAttribute('color', new THREE.BufferAttribute(cols,3));
        if(cloud){ scene.remove(cloud); cloud.geometry.dispose(); }
        cloud=new THREE.Points(g, new THREE.PointsMaterial({ size:0.012, vertexColors:true }));
        scene.add(cloud);
      }
    }
  }catch(e){}
}
async function pollStatus(){
  try{
    const s=await (await fetch('/status')).json();
    el('frame').textContent=(s.frame??'-')+' / '+(s.num_frames??'-');
    el('gauss').textContent=(s.num_gaussians??'-').toLocaleString?.(s.num_gaussians?.toLocaleString?.()) ?? (s.num_gaussians??'-');
    el('pts').textContent=(s.preview_points??'-');
    const g=s.guidance||{};
    const cov=(g.coverage!=null)? g.coverage : null;
    el('cov').textContent = cov!=null ? (Math.round(cov*100)+'%') : '-';
    el('covbar').style.width = cov!=null ? (Math.round(cov*100)+'%') : '0%';
    if(g.hints && g.hints.length){
      el('hints').textContent = 'Next views to capture ('+g.hints.length+'):\\n' +
        g.hints.slice(0,4).map(h=>'  az '+Math.round(h.azimuth_deg)+'\\u00b0 / el '+Math.round(h.elevation_deg)+'\\u00b0').join('\\n');
    }
  }catch(e){}
}
setInterval(pollSnapshot, 700);
setInterval(pollStatus, 600);

function animate(){ requestAnimationFrame(animate); controls.update(); renderer.render(scene,camera); }
animate();
window.addEventListener('resize',()=>{
  camera.aspect=window.innerWidth/window.innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>
"""
