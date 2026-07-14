"""
tests/smoke_122gu_rename_snapshot.py
====================================

Patch 122gu — snapshot before a marker rename, so Save model is undoable.

rename_markers rewrites project.toml and the pose files IN PLACE. Before any
write it now copies project.toml and every pose file about to be rewritten
into backups/marker_rename-<run_id>/ (house run-id format), with a
manifest.toml recording the rename map and each file's origin — so an undo
is a plain copy back.

* snapshot_before_rename(config_path, pose_files, rename_map) -> snapshot dir
* rename_markers(..., snapshot=True) by default; snapshot=False opts out;
  dry_run never snapshots. The path is returned as summary["snapshot_dir"]
  and surfaced in the form's confirm/success dialogs.

Checks include a real round-trip: snapshot -> rename -> restore restores the
original project.toml byte-for-byte.
"""
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

P = T = 0


def check(label, cond, *, detail=""):
    global P, T
    T += 1
    if cond:
        P += 1
    else:
        print(f"FAIL: {label}" + (f" — {detail}" if detail else ""))


def _project(tmp: Path):
    from mufasa.project_layout import (
        PROJECT_LAYOUT_VERSION,
        write_project_toml,
        write_skeleton,
    )
    cp = tmp / "project.toml"
    bps = ["nose", "ear_left", "ear_right"]
    write_project_toml(cp, {
        "project_layout_version": PROJECT_LAYOUT_VERSION,
        "pose": {"body_parts": bps, "file_type": "parquet"},
    })
    write_skeleton(cp, nodes=bps,
                   edges=[("nose", "ear_left"), ("ear_left", "ear_right")])
    return cp


def main():
    import shutil

    from mufasa.model.marker_rename import rename_markers, snapshot_before_rename
    from mufasa.project_layout import project_paths_from_config, read_project_toml

    check("snapshot_before_rename is exported",
          callable(snapshot_before_rename))

    tmp = Path(tempfile.mkdtemp())
    cp = _project(tmp)
    # fake pose files (bytes only — the snapshot is a byte copy)
    ipd = Path(project_paths_from_config(cp)["input_pose_dir"])
    ipd.mkdir(parents=True, exist_ok=True)
    for n in ("vid1.parquet", "vid2.parquet"):
        (ipd / n).write_bytes(b"ORIGINAL-" + n.encode())
    pose_files = [str(ipd / "vid1.parquet"), str(ipd / "vid2.parquet")]

    snap = Path(snapshot_before_rename(cp, pose_files, {"ear_left": "left_ear"}))
    check("snapshot dir created under backups/ with a run-id name",
          snap.parent.name == "backups"
          and snap.name.startswith("marker_rename-"))
    check("project.toml + every pose file copied",
          (snap / "project.toml").exists()
          and (snap / "input_pose" / "vid1.parquet").exists()
          and (snap / "input_pose" / "vid2.parquet").exists())
    man = snap / "manifest.toml"
    parsed = tomllib.loads(man.read_text()) if man.exists() else {}
    check("manifest.toml is valid TOML with the rename map + files",
          parsed.get("rename_map") == {"ear_left": "left_ear"}
          and len(parsed.get("files", [])) == 2)
    check("snapshot copies are byte-identical to the originals",
          (snap / "input_pose" / "vid1.parquet").read_bytes() == b"ORIGINAL-vid1.parquet")

    # dry run must not snapshot
    tmp2 = Path(tempfile.mkdtemp())
    cp2 = _project(tmp2)
    d = rename_markers(cp2, {"ear_left": "left_ear"}, dry_run=True)
    check("dry_run takes no snapshot (snapshot_dir is None)",
          d.get("snapshot_dir") is None and not (tmp2 / "backups").exists())

    # real run snapshots by default and restore works (no pose files -> toml only)
    before = cp2.read_text()
    res = rename_markers(cp2, {"ear_left": "left_ear"})
    sdir = res.get("snapshot_dir")
    ok_snap = bool(sdir) and Path(sdir).exists()
    renamed = read_project_toml(cp2)["pose"]["body_parts"] == ["nose", "left_ear", "ear_right"]
    shutil.copy2(Path(sdir) / "project.toml", cp2)
    check("default run snapshots, renames, and the snapshot restores exactly",
          ok_snap and renamed and cp2.read_text() == before)

    # opt-out
    tmp3 = Path(tempfile.mkdtemp())
    cp3 = _project(tmp3)
    r3 = rename_markers(cp3, {"ear_left": "left_ear"}, snapshot=False)
    check("snapshot=False opts out", r3.get("snapshot_dir") is None)

    form = (REPO / "mufasa" / "ui_qt" / "forms" / "model_modifications.py").read_text()
    check("form surfaces the snapshot in confirm + success dialogs",
          "backups/" in form and "snapshot_dir" in form)

    print(f"smoke_122gu_rename_snapshot: {P}/{T} checks passed")
    return 0 if P == T else 1


if __name__ == "__main__":
    sys.exit(main())
