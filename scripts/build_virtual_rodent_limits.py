#!/usr/bin/env python3
"""Build ``mufasa/assets/models/pose3d/virtual_rodent/virtual_rodent_limits.json``
from DeepMind's virtual rodent (dm_control ``rodent.xml`` and its skin,
copied next to it; Apache 2.0).

For each of the project's rat labels: the highest it can be (a hind paw on
the floor, every joint within range, no skin below the floor) and the
lowest (no skin below the floor), found by a seeded random search over the
joint angles and the body's pitch. For the spine: the range of the up/down
bend of the lumbar region (back_V2→back_L6 to back_T8→back_T4) and of the
neck (back_T8→back_T4 to back_T4→head_back), over the spine joints sampled
uniformly within range. See ``mufasa.pose3d.rodent_model``.

    python scripts/build_virtual_rodent_limits.py [--samples 200000] [--seed 0]
"""
from __future__ import annotations

import argparse

from mufasa.pose3d.rodent_model import LIMITS_PATH, build_limits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(LIMITS_PATH))
    a = ap.parse_args()
    build_limits(samples=a.samples, seed=a.seed, out=a.out)


if __name__ == "__main__":
    main()
