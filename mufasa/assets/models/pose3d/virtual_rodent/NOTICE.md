# Virtual rodent (DeepMind dm_control)

`rodent.xml` (MuJoCo MJCF model of a rat) and `rodent_walker_skin.skn` (its
skinned surface mesh) are copied unchanged from DeepMind's dm_control
repository, `dm_control/locomotion/walkers/assets/`, at commit
d7e52e3602a555df6cc20cc928ea5b41e9920158 (https://github.com/google-deepmind/dm_control).

Copyright 2020 The dm_control Authors. Licensed under the Apache License,
Version 2.0 (see `LICENSE` in this folder).

Reference: Merel J, Aldarondo D, Marshall J, Tassa Y, Wayne G, Ölveczky B.
"Deep neuroethology of a virtual rodent." ICLR 2020.

Mufasa uses the model only for anatomical limits (joint ranges, segment
lengths, skin surface) in `mufasa.pose3d.rodent_model`.
