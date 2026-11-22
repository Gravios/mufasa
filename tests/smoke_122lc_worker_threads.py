"""Smoke test for patch 122lc — one thread per 3-D inference worker.

User: 39 videos on 8 workers ran for 14 hours (10 done, "2500 minutes left"); top showed load
average 118, each worker at 100–270 % CPU, the CPU 98.6 % busy: "fix the code."

* every worker of the per-video pool (``pipeline.run_pose3d``) and of the per-chunk pool
  (``mono._pool``) runs PyTorch and every BLAS / OpenMP pool with one thread, forked or spawned,
  while the parent keeps its own settings;
* a worker that dies ends the run with an error naming how many videos are done, where they
  are, and what to do.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def probe(_=None):
    from mufasa.pose3d.threads import thread_counts
    return os.getpid(), thread_counts()


def fake_job(args):
    job, settings, out_dir = args
    return job.stem, {"threads": probe()[1], "pid": os.getpid()}


def dying_job(args):
    job, settings, out_dir = args
    if job.stem.endswith("2"):
        os._exit(1)
    return job.stem, {}


def main() -> int:
    import torch

    from mufasa.pose3d import mono, pipeline
    from mufasa.pose3d.threads import thread_counts, worker_pool
    torch.set_num_threads(max(2, os.cpu_count() or 2))
    parent = thread_counts()

    def all_one(tc):
        return bool(tc) and all(v == 1 for v in tc.values())

    for name, ctx in (("forked", None), ("spawned", multiprocessing.get_context("spawn"))):
        with worker_pool(2, ctx) as ex:
            res = list(ex.map(probe, range(4)))
        pids = {p for p, _ in res}
        check(f"worker pool, {name}: {len(pids)} workers, each {res[0][1]}", all(all_one(tc) for _, tc in res))
    check(f"the parent keeps its own settings ({parent.get('torch')} PyTorch threads)",
          thread_counts().get("torch") == parent.get("torch") and parent.get("torch", 0) >= 2)

    pool = mono._pool(2, "cpu")
    try:
        res = list(pool.map(probe, range(4)))
    finally:
        pool.shutdown()
    check(f"per-chunk pool (one video on several workers): {res[0][1]}", all(all_one(tc) for _, tc in res))

    # the per-video pool, through run_pose3d
    jobs = [pipeline.Job(stem=f"v{i}", pose=Path("x"), camera=Path("c")) for i in range(4)]
    settings = pipeline.RunSettings(workers=2)
    real = pipeline._run_job_quiet
    with tempfile.TemporaryDirectory() as td:
        try:
            pipeline._run_job_quiet = fake_job
            said = []
            reps = pipeline.run_pose3d(jobs, settings, Path(td) / "run", progress=said.append)
        finally:
            pipeline._run_job_quiet = real
    tcs = [r["threads"] for r in reps.values()]
    check(f"per-video pool (run_pose3d, 4 videos on 2 workers): every worker {tcs[0]}",
          len(tcs) == 4 and all(all_one(tc) for tc in tcs) and any("4 videos on 2 workers" in s for s in said))

    # a worker that dies
    with tempfile.TemporaryDirectory() as td:
        err = ""
        try:
            pipeline._run_job_quiet = dying_job
            pipeline.run_pose3d(jobs, settings, Path(td) / "run")
        except RuntimeError as e:
            err = str(e)
        except Exception as e:  # noqa: BLE001
            err = f"other: {type(e).__name__}: {e}"
        finally:
            pipeline._run_job_quiet = real
    check(f"a worker that dies: an error saying what to do ({err[:90]}…)",
          "worker process died" in err and "fewer workers" in err and "videos done" in err)

    doc = (REPO / "mufasa" / "pose3d" / "threads.py").read_text()
    check("why, written down (load 118, one thread per worker)", "load average 118" in doc)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122lc_worker_threads: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
