"""
mufasa.biomech
==============

A biomechanical model of the rat, built on DeepMind's virtual rodent in MuJoCo, to go with
the marker and silhouette fits.

1. The legs bend to the gait's paws (patch 122lh; :mod:`mufasa.pose3d.leg_ik`).
2. Joint torques from the observed motion (patch 122li). The model is scaled to the animal
   (:mod:`.mjmodel`). All of its joints are fitted to the 3-D labels and the gait's paws
   (:mod:`.fit`). Inverse dynamics then gives the generalised forces, and ground forces at the
   planted paws explain the whole-body balance (:mod:`.dynamics`).
3. Muscles: paths, moment arms, and loads shared out to produce those torques (planned;
   hindlimbs first: Eng et al. 2008, Johnson et al. 2008).
4. Simulation: trajectories optimised for tracking and effort (planned).

MuJoCo is an optional dependency: ``pip install "mufasa[biomech]"``.
"""
