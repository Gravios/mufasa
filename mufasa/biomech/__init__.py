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
3. The hindlimbs' muscles (patch 122lj): read from an OpenSim rat hindlimb model (Johnson et
   al. 2008) and carried onto the virtual rodent's bones (:mod:`.osim`; Eng et al. 2008's
   architecture as an option), and their forces shared out to produce those torques by static
   optimisation (:mod:`.muscle`).
4. Whole movements re-solved (patch 122lk): every joint, the paws' forces and the muscles'
   activations over windows of frames, for tracking, balance and effort, the activations smooth
   in time (:mod:`.optimise`).

MuJoCo is an optional dependency: ``pip install "mufasa[biomech]"``.
"""
