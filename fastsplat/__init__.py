"""fastsplat: fast object-centric Gaussian splatting from sparse images.

Pipeline (SplaTAM-free):

    ZED -> ROS2 (Zenoh transfer)  ->  ingest sparse keyframes
                                  ->  MapAnything feed-forward SfM (poses + metric point cloud)
                                  ->  object isolation (auto 3D crop or SAM2)
                                  ->  gsplat time-boxed optimization
                                  ->  object splat (.ply)

The goal is a good splat of a *single object* from sparse views in <= 5 minutes.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
