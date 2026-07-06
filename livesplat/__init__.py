"""Live-splat operator tooling.

This package holds everything the *operator* touches, kept deliberately
separate from the SLAM core so it can evolve (or be swapped for a different
front-end) without churning ``scripts/zed2i_splat_live.py``:

* :mod:`livesplat.roi`             -- region-of-interest data structures.
* :mod:`livesplat.bridge`          -- thread-safe hand-off between the SLAM
                                      loop and the viewer web server.
* :mod:`livesplat.viewer_server`   -- browser viewer (HTTP + SSE) + ROI POST.
* :mod:`livesplat.capture_advisor` -- viewpoint-coverage analysis and
                                      next-best-view guidance for high-quality
                                      object captures.

Named ``livesplat`` (not ``operator``) on purpose: ``operator`` is a Python
standard-library module and shadowing it breaks unrelated imports.
"""

from livesplat.roi import RegionOfInterest, RefineRequest  # noqa: F401
from livesplat.bridge import LiveSplatBridge  # noqa: F401
