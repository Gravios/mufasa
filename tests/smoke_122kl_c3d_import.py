"""Smoke test for patch 122kl — motion capture (C3D) into 3-D pose.

User: "I would like for it to be able to import c3d files from motion
capture".

C3D files are written here (a small writer below) in every storage form
the reader handles: integer and float data; Intel, MIPS (big-endian) and
DEC (VAX float) processors; labels continued in LABELS2; units in metres;
invalid points; a frame count beyond the 16-bit header (POINT:FRAMES);
POINT:DATA_START = 0 (header used); an empty character parameter.
(The reader was also checked against ezc3d and the ``c3d`` package on
vendor files from Vicon, Qualisys, Cortex, BTS, Kistler, Optotrak and
Motion Analysis: identical points, labels, rates, residuals.)

Then: millimetres and z up (a y-up file turned), renaming, skeleton lines,
the 3-D pose file and its metadata, the CLI into a project's
sources/pose3d/, the tab's import section, the 3-D viewer finding the
files (with no video), and the section's registration.
"""
from __future__ import annotations

import os
import struct
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


INTEL, DEC, MIPS = 84, 85, 86


def _f32(v, proc):
    v = np.asarray(v, np.float32)
    if proc == MIPS:
        return v.astype(">f4").tobytes()
    if proc == DEC:
        b = (v * 4).astype("<f4").view(np.uint8).reshape(-1, 4)[:, [2, 3, 0, 1]]
        return b.tobytes()
    return v.astype("<f4").tobytes()


def _i16(v, proc):
    return np.asarray(v, np.int64).astype(">i2" if proc == MIPS else "<i2").tobytes()


def write_c3d(path, X, invalid, labels, *, rate=100.0, fmt="float", proc=INTEL, units="mm",
              scale=0.1, split_labels=None, header_frames=None, data_start_param=True,
              empty_char_param=False):
    """Minimal C3D writer (3-D points only)."""
    T, M, _ = X.shape
    s = -abs(scale) if fmt == "float" else abs(scale)
    e = ">" if proc == MIPS else "<"

    def rec_group(gid, name):
        nb = name.encode()
        return struct.pack("bb", len(nb), -gid) + nb + struct.pack(e + "h", 3) + b"\x00"

    def rec_param(gid, name, dtype, dims, data):
        nb = name.encode()
        body = struct.pack("bB", dtype, len(dims)) + bytes(dims) + data + b"\x00"
        return struct.pack("bb", len(nb), gid) + nb + struct.pack(e + "h", 2 + len(body)) + body

    def chars(strings, width):
        return b"".join(x.ljust(width).encode()[:width] for x in strings)

    parts = [rec_group(1, "POINT"), rec_group(2, "TRIAL")]
    lab1, lab2 = (labels, []) if split_labels is None else (labels[:split_labels], labels[split_labels:])
    parts.append(rec_param(1, "USED", 2, [], _i16([M], proc)))
    parts.append(rec_param(1, "SCALE", 4, [], _f32([s], proc)))
    parts.append(rec_param(1, "RATE", 4, [], _f32([rate], proc)))
    parts.append(rec_param(1, "FRAMES", 2, [], _i16([T if T < 32768 else T - 65536], proc)))
    parts.append(rec_param(1, "UNITS", -1, [len(units)], units.encode()))
    parts.append(rec_param(1, "LABELS", -1, [16, len(lab1)], chars(lab1, 16)))
    if lab2:
        parts.append(rec_param(1, "LABELS2", -1, [16, len(lab2)], chars(lab2, 16)))
    if empty_char_param:
        parts.append(rec_param(1, "DESCRIPTIONS", -1, [0, 0], b""))
    nparam_blocks = 2
    data_start = 2 + nparam_blocks
    parts.append(rec_param(1, "DATA_START", 2, [], _i16([data_start if data_start_param else 0], proc)))
    body = b"".join(parts) + b"\x00\x00"
    psec = bytes([1, 0x50, nparam_blocks, proc]) + body
    psec = psec.ljust(512 * nparam_blocks, b"\x00")
    last = header_frames if header_frames is not None else T
    hdr = bytearray(512)
    hdr[0], hdr[1] = 2, 0x50
    hdr[2:4] = _i16([M], proc)
    hdr[4:6] = _i16([0], proc)
    hdr[6:8] = _i16([1], proc)
    hdr[8:10] = _i16([last], proc)
    hdr[12:16] = _f32([s], proc)
    hdr[16:18] = _i16([data_start], proc)
    hdr[20:24] = _f32([rate], proc)
    resid = np.where(invalid, -1, (1 << 8) | 7)                          # camera bit 1, residual 7 × |scale|
    if fmt == "float":
        w = np.concatenate([X, resid[..., None]], 2).astype(np.float32)
        data = _f32(w.ravel(), proc)
    else:
        q = np.rint(X / abs(scale)).astype(np.int64)
        w = np.concatenate([q, resid[..., None]], 2)
        data = _i16(w.ravel(), proc)
    Path(path).write_bytes(bytes(hdr) + psec + data)


def main() -> int:
    from mufasa.pose3d.c3d import c3d_to_pose3d, import_c3d, parse_edges, parse_rename, read_c3d
    from mufasa.pose3d.io import read_pose3d

    rng = np.random.default_rng(0)
    T, M = 120, 5
    labels = ["head_nose", "head_back", "back_T4", "back_L6", "tail_V6"]
    t = np.arange(T) / 100
    X = np.stack([np.c_[200 * np.sin(t + k), 150 * np.cos(t + k), 50 + 20 * k + 10 * np.sin(3 * t)]
                  for k in range(M)], 1)
    invalid = rng.random((T, M)) < 0.1

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cases = {
            "float, Intel": dict(fmt="float", proc=INTEL),
            "integer, Intel": dict(fmt="int", proc=INTEL, scale=0.05),
            "float, MIPS (big-endian)": dict(fmt="float", proc=MIPS),
            "integer, MIPS (big-endian)": dict(fmt="int", proc=MIPS, scale=0.05),
            "float, DEC (VAX floats)": dict(fmt="float", proc=DEC),
            "integer, DEC": dict(fmt="int", proc=DEC, scale=0.05),
        }
        for name, kw in cases.items():
            f = td / "c.c3d"
            write_c3d(f, X, invalid, labels, **kw)
            c = read_c3d(f)
            tol = 0.05 / 2 + 1e-6 if kw["fmt"] == "int" else 1e-3
            ok_xyz = np.allclose(np.nan_to_num(c.points), np.where(invalid[..., None], 0, X), atol=tol)
            check(f"{name}: points, invalid markers, labels, rate",
                  ok_xyz and (np.isnan(c.points[..., 0]) == invalid).all() and c.labels == labels
                  and abs(c.rate - 100) < 1e-4 and c.processor == {INTEL: "intel", MIPS: "mips", DEC: "dec"}[kw["proc"]])
        check("residuals: low byte × |scale| (7 × 0.1 mm)",
              np.allclose(read_c3d(td / "c.c3d").residuals[~invalid], 7 * 0.05))

        f = td / "split.c3d"
        many = [f"m{i}" for i in range(M)]
        write_c3d(f, X, invalid, many, split_labels=3)
        check("labels continued in LABELS2", read_c3d(f).labels == many)
        write_c3d(f, X, invalid, ["a", "a", "", "b", "c"])
        check("duplicate / empty labels made unique", read_c3d(f).labels == ["a", "a_1", "M003", "b", "c"])
        write_c3d(f, X, invalid, labels, header_frames=40)
        check("POINT:FRAMES beyond the header's frame count", len(read_c3d(f).points) == T)
        write_c3d(f, X, invalid, labels, data_start_param=False, empty_char_param=True)
        check("POINT:DATA_START = 0 → header; empty character parameter skipped",
              len(read_c3d(f).points) == T and read_c3d(f).labels == labels)

        # metres, y up → mm, z up
        Xm = X[..., [0, 2, 1]] * np.array([1, 1, -1]) / 1000.0         # same motion with y up, in m
        f = td / "yup.c3d"
        write_c3d(f, Xm, invalid, labels, units="m", scale=0.0001)
        c = read_c3d(f)
        df, meta = c3d_to_pose3d(c, up="y", source=str(f))
        got = np.stack([df[[m + "_x", m + "_y", m + "_z"]].to_numpy() for m in labels], 1)
        check("metres → millimetres, y up → z up (same motion)",
              np.allclose(np.nan_to_num(got), np.where(invalid[..., None], 0, X), atol=0.05))
        check("likelihood 1 measured / 0 missing; residual column",
              (df["head_nose_likelihood"].to_numpy() == (~invalid[:, 0]).astype(float)).all()
              and "head_nose_residual_mm" in df)
        check("metadata: mm, fps, box from the data, rat skeleton for known names",
              meta["units"] == "mm" and meta["fps"] == 100 and meta["box_z"][0] <= 0
              and ["head_nose", "head_back"] not in meta["edges"] and ["head_back", "back_T4"] in meta["edges"])
        df2, meta2 = c3d_to_pose3d(read_c3d(td / "c.c3d"), rename=parse_rename("head_nose=nose, tail_V6=tail"),
                                   edges=parse_edges("nose-head_back, back_L6-tail"), frames=(10, 50))
        check("rename, skeleton lines, frame range",
              "nose_x" in df2 and "tail_x" in df2 and meta2["edges"] == [["nose", "head_back"], ["back_L6", "tail"]]
              and len(df2) == 40 and df2["c3d_frame"].iloc[0] == 11)

        out = import_c3d(td / "c.c3d", td / "out")
        d3, m3 = read_pose3d(out)
        check("import_c3d writes <stem>_3d.parquet with the mufasa_pose3d metadata",
              out.name == "c_3d.parquet" and "motion capture (C3D" in m3["method"] and len(d3) == T)

        # project: CLI, tab, viewer
        from mufasa.project_layout import PROJECT_CONFIG_FILENAME, ProjectPaths, write_project_toml
        root = td / "proj"
        root.mkdir()
        ProjectPaths(root).ensure_skeleton()
        write_project_toml(root / PROJECT_CONFIG_FILENAME, {"project_layout_version": 1, "project_name": "c3d"})
        cfg = str(root / PROJECT_CONFIG_FILENAME)
        from mufasa.tools.pose3d_cli import main as cli
        code = cli(["import-c3d", str(td / "yup.c3d"), "--project", cfg, "--up", "y"])
        check("CLI import-c3d → sources/pose3d/", code == 0 and (root / "sources" / "pose3d" / "yup_3d.parquet").is_file())

        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        from mufasa.ui_qt.forms.pose3d import C3DImportForm, Pose3DViewerLauncher
        form = C3DImportForm(config_path=cfg)
        form.files.setText(str(td / "c.c3d"))
        form.rename.setText("head_nose=nose")
        args = form.collect_args()
        form.target(**args)
        check("3-D tab: import section writes sources/pose3d/",
              (root / "sources" / "pose3d" / "c_3d.parquet").is_file() and args["rename"] == {"head_nose": "nose"})
        lau = Pose3DViewerLauncher(config_path=cfg)
        items = [lau.file.itemText(i) for i in range(lau.file.count())]
        check("3-D viewer section: '(no video)' lists the imported motion capture",
              lau.video.currentText() == "(no video)" and "pose3d/c_3d.parquet" in items and lau.open_btn.isEnabled())
        from mufasa.ui_qt.pose3d_viewer import find_pose3d, load_pose3d
        check("viewer finds motion capture beside a video of the same name",
              find_pose3d(str(root / "sources" / "videos" / "yup.mp4"), cfg) == root / "sources" / "pose3d" / "yup_3d.parquet")
        p = load_pose3d(root / "sources" / "pose3d" / "c_3d.parquet")
        check("viewer loads it (markers, frames, box)", len(p.names) == M and p.box is not None)

    from mufasa.section_provenance import SECTIONS
    s = SECTIONS.get("pose3d_c3d")
    check("section registered: 3-D / Import motion capture (C3D), detect sources/pose3d",
          s is not None and s.page == "3-D" and s.detect_path(Path("/r")) == Path("/r/sources/pose3d"))
    page = (REPO / "mufasa" / "ui_qt" / "pages" / "pose3d_page.py").read_text()
    check("tab: section added", 'page.add_section("Import motion capture (C3D)", [(C3DImportForm, {})])' in page)
    cli_src = (REPO / "mufasa" / "tools" / "pose3d_cli.py").read_text()
    check("CLI does not import Qt", "ui_qt" not in cli_src)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122kl_c3d_import: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
