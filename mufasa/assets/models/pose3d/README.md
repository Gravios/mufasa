# 3-D pose models

## `rat7m_trunk_prior.npz`

Posture statistics of freely moving rats used by single-camera 3-D
inference (`mufasa.pose3d.mono`) to set the heights of the spine and hip
markers, the lumbar (pelvis) angle and the limits on how the back may bend.

**Source:** Rat7M motion capture — Dunn, T. W., Marshall, J. D., Severson,
K. S., Aldarondo, D. E., Hildebrand, D. G. C., Chettih, S. N., Wang, W. L.,
Gellis, A. J., Carlson, D. E., Aronov, D., Freiwald, W. A., Wang, F. &
Ölveczky, B. P. (2021). *Geometric deep learning enables 3D kinematic
profiling across species and environments.* Nature Methods 18, 564–573.
Data: Figshare item 13739233, "Motion Capture Data and Camera Parameters"
(Rat 7M collection). Licensed **CC BY 4.0**; this file is a derived,
subsampled table and keeps that license and attribution.

**Contents:** 80 000 frames sampled from all seven sessions (five rats):

| array | meaning |
|---|---|
| `features` | head height (HeadB z, mm), head pitch (HeadB→HeadF, deg), horizontal HeadB→hip-midpoint distance (mm) |
| `targets` | heights (mm) of SpineF, SpineM, SpineL and the hip midpoint |
| `lumbar` | SpineL→SpineM elevation (deg) |
| `roll` | head roll from the HeadL marker (deg) |
| `subject` | Rat7M subject id |
| `bend_spineM`, `bend_spineF` | 5/50/95th percentiles of the spine bend (deg, + = upward/backward) at SpineM and SpineF for head-height bins `bend_edges` |

**Rebuild:** `python scripts/build_rat7m_prior.py /path/to/mocap_dir`
(reads `mocap-s*-d*.mat`).

Rat7M markers sit on a head implant and on pierced skin markers, so a
project's labels differ from them; `mufasa.pose3d.mono.MonoConfig` holds
the mapping (head-height and head-pitch offsets, where each spine label
sits between the Rat7M spine markers).
