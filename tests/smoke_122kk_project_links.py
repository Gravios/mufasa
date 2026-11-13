"""Smoke test for patch 122kk — project links that survive an sshfs mount.

User: "I am loading locally with sshfs from the server. The links break
under this condition."

The server's ``/`` is simulated by a folder (``server/``); the project's
video links hold server paths (``/proc/videos/K7.mp4``), which do not
exist here, exactly as on an sshfs mount.

* a broken link is found through the mount offset and made relative; the
  relative link means the same file on the server (checked by resolving it
  from the project's server path);
* a working absolute link becomes relative to the same target;
* a link whose target is not reachable here (or only through a mount that
  is not an ancestor of the project) is reported, not touched;
* --dry-run changes nothing; a second run has nothing to do;
* new imports link relatively; the workbench repairs on load; CLI entry
  point; the classifier table no longer disconnects an unconnected signal.
"""
from __future__ import annotations

import os
import posixpath
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

checks: list[tuple[str, bool]] = []


def check(name: str, ok: bool) -> None:
    checks.append((name, bool(ok)))


def main() -> int:
    from mufasa.utils.project_links import link_file, repair_project_links
    from mufasa.utils.project_links import main as relink_main

    with tempfile.TemporaryDirectory() as td:
        td = Path(td).resolve()
        mount = td / "cloud" / "server"                  # the server's "/" mounted here
        proj_server = Path("/proc/mufasa/P/P")           # the project's path on the server
        proj = mount / proj_server.relative_to("/")
        vids = proj / "sources" / "videos"
        vids.mkdir(parents=True)
        (mount / "proc" / "videos").mkdir(parents=True)
        for n in ("K7", "K8"):
            (mount / "proc" / "videos" / f"{n}.mp4").write_bytes(b"video")
            os.symlink(f"/proc/videos/{n}.mp4", vids / f"{n}.mp4")       # server path: broken here
        os.symlink("/proc/videos/missing.mp4", vids / "gone.mp4")         # not on the server either
        local_real = td / "elsewhere" / "K9.mp4"
        local_real.parent.mkdir()
        local_real.write_bytes(b"video")
        os.symlink(local_real, vids / "K9.mp4")                          # absolute, working here
        (proj / "project.toml").write_text("project_layout_version = 1\n")

        check("setup: K7/K8 links broken here, as on sshfs", not (vids / "K7.mp4").exists())
        dry = repair_project_links(proj, dry_run=True)
        check("dry run: reports, changes nothing",
              len(dry.relinked) == 3 and os.readlink(vids / "K7.mp4") == "/proc/videos/K7.mp4")

        rep = repair_project_links(proj)
        sp, lp = rep.offset
        check(f"mount offset found ({sp} → {lp})",
              Path(lp) / Path("/proc/videos/K7.mp4").relative_to(sp) == mount / "proc" / "videos" / "K7.mp4")
        check("broken links now resolve here", (vids / "K7.mp4").exists() and (vids / "K8.mp4").exists())
        rel = os.readlink(vids / "K7.mp4")
        on_server = posixpath.normpath(posixpath.join(str(proj_server / "sources" / "videos"), rel))
        check(f"the relative link names the same file on the server ({rel} → {on_server})",
              not os.path.isabs(rel) and on_server == "/proc/videos/K7.mp4")
        check("working absolute link: same target, relative",
              not os.path.isabs(os.readlink(vids / "K9.mp4"))
              and (vids / "K9.mp4").resolve() == local_real)
        check("unreachable target reported and left alone",
              [Path(link).name for link, _ in rep.broken] == ["gone.mp4"]
              and os.readlink(vids / "gone.mp4") == "/proc/videos/missing.mp4"
              and "mufasa-relink" in rep.text())
        again = repair_project_links(proj)
        check("second run: nothing left to change", not again.relinked and len(again.broken) == 1)

        # a target reachable only through some other mount (not under the
        # project's ancestors) is reported, not guessed
        alone = td / "home" / "P"
        (alone / "sources" / "videos").mkdir(parents=True)
        os.symlink("/srv/videos/K7.mp4", alone / "sources" / "videos" / "K7.mp4")
        (td / "srvmount" / "videos").mkdir(parents=True)
        (td / "srvmount" / "videos" / "K7.mp4").write_bytes(b"v")
        rep2 = repair_project_links(alone)
        check("target only under another mount: reported, left alone",
              not rep2.relinked and len(rep2.broken) == 1
              and os.readlink(alone / "sources" / "videos" / "K7.mp4") == "/srv/videos/K7.mp4")

        # imports
        src = td / "data" / "v.mp4"
        src.parent.mkdir()
        src.write_bytes(b"v")
        lk = link_file(src, vids / "v.mp4")
        check("import: relative link to the video", not os.path.isabs(os.readlink(lk)) and lk.resolve() == src)

        # CLI
        os.symlink("/proc/videos/K8.mp4", vids / "K8b.mp4")
        code = relink_main([str(proj / "project.toml")])
        check("mufasa-relink: relinks, exit 1 while a link stays broken",
              (vids / "K8b.mp4").exists() and code == 1)

    rw = (REPO / "mufasa" / "utils" / "read_write.py").read_text()
    check("video import (single and many) links relatively",
          rw.count("link_file(") == 2 and "os.symlink(source_path, destination)" not in rw)
    check("frame-folder import links relatively",
          "link_file(source, destination)" in (REPO / "mufasa" / "ui_qt" / "forms" / "video_frames.py").read_text())
    wb = (REPO / "mufasa" / "ui_qt" / "workbench_app.py").read_text()
    check("workbench repairs links when a project opens",
          "_repair_links(project_config_path)" in wb and "repair_project_links" in wb)
    check("entry point mufasa-relink",
          'mufasa-relink    = "mufasa.utils.project_links:main"' in (REPO / "pyproject.toml").read_text())
    clf = (REPO / "mufasa" / "ui_qt" / "forms" / "classifier.py").read_text()
    check("classifier table: double-click connected once (no disconnect warning)",
          "cellDoubleClicked.disconnect()" not in clf and "_dbl_click_connected" in clf)

    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
    n_ok = sum(ok for _, ok in checks)
    print(f"smoke_122kk_project_links: {n_ok}/{len(checks)} checks passed")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
