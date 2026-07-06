import json
import struct
import urllib.request

import numpy as np
import pytest

from livesplat.bridge import LiveSplatBridge
from livesplat.viewer_server import OperatorViewerServer


@pytest.fixture()
def server():
    bridge = LiveSplatBridge(max_preview_points=5000)
    srv = OperatorViewerServer(bridge, host="127.0.0.1", port=0).start()
    # port=0 -> OS assigns a free port; read it back from the bound socket.
    srv.port = srv._httpd.server_address[1]
    yield bridge, srv, f"http://127.0.0.1:{srv.port}"
    srv.stop()


def _get(url):
    return urllib.request.urlopen(url, timeout=5).read()


def test_index_page_served(server):
    _, _, base = server
    html = _get(base + "/").decode()
    assert "Live Splat Operator" in html and "three" in html


def test_snapshot_binary_roundtrip(server):
    bridge, _, base = server
    bridge.publish_snapshot(np.random.randn(8000, 3).astype(np.float32),
                            np.random.rand(8000, 3).astype(np.float32))
    v = json.loads(_get(base + "/snapshot_version"))["version"]
    assert v >= 1
    blob = _get(base + "/snapshot.bin")
    n = struct.unpack("<I", blob[:4])[0]
    assert 0 < n <= 5000
    assert len(blob) == 4 + n * 12 + n * 3


def test_status_reports_published_fields(server):
    bridge, _, base = server
    bridge.publish_status(frame=7, num_frames=45, num_gaussians=1234)
    st = json.loads(_get(base + "/status"))
    assert st["frame"] == 7 and st["num_gaussians"] == 1234


def test_roi_post_enqueues_request(server):
    bridge, _, base = server
    body = json.dumps({"roi": {"min_xyz": [0, 0, 0], "max_xyz": [1, 1, 1], "label": "op"},
                       "quality": 2.5}).encode()
    req = urllib.request.Request(base + "/roi", data=body,
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
    assert resp["request_id"] == 1
    r = bridge.pop_refine_request()
    assert r.quality == 2.5 and r.roi.label == "op"


def test_empty_snapshot_before_any_publish(server):
    _, _, base = server
    blob = _get(base + "/snapshot.bin")
    assert struct.unpack("<I", blob[:4])[0] == 0
