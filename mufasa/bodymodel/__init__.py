"""
mufasa.bodymodel
================

Patch 122kq: fit a rigged 3-D body model (a Blender armature with a skinned
mesh, e.g. the HBP mouse) to top-down video, frame by frame, from the
animal's silhouette and the 2-D keypoints, with joints kept within
per-joint limits; refined over time forwards and backwards; optional
shaded overlay video.

* :mod:`~mufasa.bodymodel.rig` — the rig as arrays (``*.rig.npz``), made
  from a ``.blend`` with Blender's Python module (``pip install bpy``),
  in a separate process.
* :mod:`~mufasa.bodymodel.fit` — posture, limits, forward kinematics,
  skinning, markers, the per-frame fit and the sequence fit.
* :mod:`~mufasa.bodymodel.render` — the overlay (painter's algorithm,
  fur colours baked per face), and the overlay video.
"""
