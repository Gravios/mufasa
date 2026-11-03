"""
mufasa.pose_importers.pose_parquet
==================================

Long-format pose parquet: the shared, dependency-light layer.

Both pose-estimation front ends that feed Mufasa now write the same tidy table,
one row per (frame, individual, bodypart)::

    frame | individual | bodypart | x | y | likelihood

* **FreeDLC** names it ``<video>.fdlc.parquet`` and puts the skeleton in a
  ``<video>.fdlc.toml`` sidecar.
* **DeepLabCut** (the modified exporter) names it ``<video>.parquet`` and
  embeds the skeleton in the parquet file itself (see *Embedded metadata*).

This module is what the importer, project creation, session ingest, the
smoother and the viewers share for that format, so it imports only the
standard library plus pyarrow/pandas (lazily) — no h5py/cv2/numba chain.

Embedded metadata
-----------------
A long pose parquet may carry its own description in the parquet key/value
metadata under the key ``b"pose"``, as UTF-8 JSON::

    {
      "version": 1,
      "bodyparts": ["head_nose", "head_mid", ...],      # node order
      "skeleton": [["head_nose", "head_mid"], ...],     # name or index pairs
      "individuals": ["single"],                         # optional
      "scorer": "DLC_resnet50_..."                       # optional
    }

``bodyparts`` and ``skeleton`` use DeepLabCut's own ``config.yaml`` keys, so a
DLC exporter can copy them across verbatim. Writing it from the exporter::

    import json, pyarrow as pa, pyarrow.parquet as pq
    table = pa.Table.from_pandas(df_long, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"pose"] = json.dumps({"version": 1,
                                "bodyparts": cfg["bodyparts"],
                                "skeleton": cfg["skeleton"]}).encode()
    pq.write_table(table.replace_schema_metadata(meta), path)

or, for a file that already exists, :func:`embed_pose_metadata` /
``python -m mufasa.pose_importers.pose_parquet FILE... --skeleton-from SRC``.

Skeleton resolution (:func:`read_pose_skeleton`)
------------------------------------------------
1. embedded ``b"pose"`` metadata in the parquet;
2. the FreeDLC ``<stem>.fdlc.toml`` sidecar next to it;
3. otherwise ``None`` — the caller falls back (project ``[skeleton]``, or
   nodes-only). :func:`resolve_skeleton` adds that project step for tools
   that are handed loose files (the viewers).
"""
from __future__ import annotations

import json
import os
import tomllib
import warnings

FDLC_SUFFIX = ".fdlc.parquet"
FDLC_SKELETON_SUFFIX = ".fdlc.toml"
POSE_METADATA_KEY = b"pose"
POSE_METADATA_VERSION = 1
LONG_REQUIRED_COLUMNS = ("frame", "bodypart", "x", "y")
LONG_COLUMNS = ("frame", "individual", "bodypart", "x", "y", "likelihood")
_WIDE_SUFFIXES = ("_x", "_y", "_p", "_likelihood")


# --------------------------------------------------------------------------- #
# Skeleton normalisation (shared by the TOML sidecar and the embedded JSON)
# --------------------------------------------------------------------------- #
def _normalize_skeleton(data: dict) -> tuple[list[str], list[tuple[str, str]]]:
    """Pull ``(nodes, edges)`` out of a FreeDLC/DLC-style mapping, tolerantly.

    * nodes: top-level ``bodyparts`` (DLC / FreeDLC key), else a ``skeleton``
      table's ``nodes``, else top-level ``nodes``;
    * edges: a top-level ``skeleton`` list (DLC / FreeDLC key), else a
      ``skeleton`` table's ``edges``, else top-level ``edges``;
    * an edge may be a name pair or an integer index pair (mapped through the
      nodes; unmappable index pairs are skipped). Anything else is ignored.
    """
    sk = data.get("skeleton")
    table = sk if isinstance(sk, dict) else {}
    nodes_src = (
        data.get("bodyparts")
        or table.get("nodes")
        or data.get("nodes")
        or []
    )
    nodes = [str(n) for n in nodes_src if str(n)]

    if isinstance(sk, list):
        raw_edges = sk
    elif isinstance(table.get("edges"), list):
        raw_edges = table["edges"]
    elif isinstance(data.get("edges"), list):
        raw_edges = data["edges"]
    else:
        raw_edges = []

    edges: list[tuple[str, str]] = []
    for e in raw_edges:
        if not isinstance(e, (list, tuple)) or len(e) < 2:
            continue
        a, b = e[0], e[1]
        if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool):
            if nodes and 0 <= a < len(nodes) and 0 <= b < len(nodes):
                edges.append((nodes[a], nodes[b]))
        else:
            edges.append((str(a), str(b)))
    return nodes, edges


# --------------------------------------------------------------------------- #
# FreeDLC sidecar
# --------------------------------------------------------------------------- #
def read_fdlc_skeleton(path: str | os.PathLike) -> dict | None:
    """Read a FreeDLC skeleton sidecar and return its nodes and edges.

    FreeDLC writes ``<stem>.fdlc.toml`` next to ``<stem>.fdlc.parquet``. The
    sidecar is resolved from the parquet path, the toml path itself, or a bare
    ``<stem>.parquet`` (-> ``<stem>.fdlc.toml``), and parsed tolerantly (see
    :func:`_normalize_skeleton`).

    :returns: ``{'nodes': [...], 'edges': [(a, b), ...], 'source': <path>}``,
        or ``None`` when the sidecar is absent or unparseable (a warning is
        emitted on parse failure) — so files that predate the sidecar still
        import nodes-only.
    """
    p = os.fspath(path)
    low = p.lower()
    if low.endswith(FDLC_SUFFIX):
        toml_path = p[: -len(FDLC_SUFFIX)] + FDLC_SKELETON_SUFFIX
    elif low.endswith(FDLC_SKELETON_SUFFIX):
        toml_path = p
    else:
        toml_path = os.path.splitext(p)[0] + FDLC_SKELETON_SUFFIX
    if not os.path.isfile(toml_path):
        return None
    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError) as exc:
        warnings.warn(
            f"Could not parse FreeDLC skeleton {toml_path}: {exc}", stacklevel=2
        )
        return None
    nodes, edges = _normalize_skeleton(data)
    if not nodes and not edges:
        return None
    return {"nodes": nodes, "edges": edges, "source": toml_path}


# --------------------------------------------------------------------------- #
# Embedded metadata
# --------------------------------------------------------------------------- #
def _schema(path):
    import pyarrow.parquet as pq
    return pq.read_schema(os.fspath(path))


def read_pose_metadata(path: str | os.PathLike) -> dict | None:
    """Return the parsed ``b"pose"`` JSON embedded in a parquet, or ``None``.

    Reads only the parquet footer. Also accepts the two keys written
    separately (``b"bodyparts"`` / ``b"skeleton"``, each JSON) for exporters
    that store them flat. A malformed value warns and yields ``None`` rather
    than failing the read — the data itself is still usable.
    """
    p = os.fspath(path)
    if not p.lower().endswith(".parquet") or not os.path.isfile(p):
        return None
    try:
        meta = _schema(p).metadata or {}
    except Exception:
        return None
    raw = meta.get(POSE_METADATA_KEY)
    try:
        if raw is not None:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else None
        flat = {}
        for key in (b"bodyparts", b"skeleton"):
            if key in meta:
                flat[key.decode()] = json.loads(meta[key].decode("utf-8"))
        return flat or None
    except (ValueError, UnicodeDecodeError) as exc:
        warnings.warn(
            f"Ignoring malformed pose metadata in {os.path.basename(p)}: {exc}",
            stacklevel=2,
        )
        return None


def read_embedded_skeleton(path: str | os.PathLike) -> dict | None:
    """Skeleton from the parquet's embedded metadata, or ``None``."""
    data = read_pose_metadata(path)
    if not data:
        return None
    nodes, edges = _normalize_skeleton(data)
    if not nodes and not edges:
        return None
    return {"nodes": nodes, "edges": edges, "source": os.fspath(path)}


def read_pose_skeleton(path: str | os.PathLike) -> dict | None:
    """Resolve a pose file's skeleton: embedded metadata, then FreeDLC sidecar.

    :returns: ``{'nodes', 'edges', 'source'}`` or ``None`` when the file
        carries no skeleton by either route.
    """
    return read_embedded_skeleton(path) or read_fdlc_skeleton(path)


def resolve_skeleton(paths, *, config=None) -> dict | None:
    """Find the skeleton that applies to a set of pose/video paths.

    Patch 122ic. Order, most specific first:

    1. ``config`` — an explicit ``project.toml`` (the user said so);
    2. each pose file's own skeleton (:func:`read_pose_skeleton`: embedded,
       then ``.fdlc.toml`` sidecar) — a pose *directory* is probed through
       the long-format parquets directly inside it;
    3. the ``project.toml`` enclosing any of the paths (walked up a bounded
       number of levels), e.g. a smoothed file under ``derived/``.

    :param paths: pose files, pose folders and/or the video; ``None`` entries
        are skipped.
    :returns: ``{'nodes', 'edges', 'source'}`` with at least one edge, or
        ``None`` — the caller keeps its own default.
    """
    from mufasa.project_layout import find_project_config, read_skeleton

    def _ok(sk):
        return sk if sk and sk.get("edges") else None

    if config:
        sk = _ok(read_skeleton(config))
        if sk:
            return {**sk, "source": os.fspath(config)}
    candidates = [os.fspath(p) for p in paths if p]
    for p in candidates:
        probes = find_long_pose_parquets(p) if os.path.isdir(p) else [p]
        for f in probes:
            sk = _ok(read_pose_skeleton(f))
            if sk:
                return sk
    for p in candidates:
        cfg = find_project_config(p)
        if cfg is not None:
            sk = _ok(read_skeleton(cfg))
            if sk:
                return {**sk, "source": os.fspath(cfg)}
    return None


# --------------------------------------------------------------------------- #
# Format detection / names
# --------------------------------------------------------------------------- #
def is_long_pose_parquet(path: str | os.PathLike) -> bool:
    """True when ``path`` is a parquet in the long pose layout.

    Decided from the schema alone (footer read): the file must have
    ``frame``, ``bodypart``, ``x`` and ``y`` columns. A wide Mufasa pose file
    (``<bp>_x`` columns), a smoothed output, or any unrelated parquet is
    rejected, so a folder can be scanned without guessing from file names.
    """
    p = os.fspath(path)
    if not p.lower().endswith(".parquet") or not os.path.isfile(p):
        return False
    try:
        names = set(_schema(p).names)
    except Exception:
        return False
    return set(LONG_REQUIRED_COLUMNS) <= names


def find_long_pose_parquets(directory: str | os.PathLike) -> list[str]:
    """Long-format pose parquets directly inside ``directory`` (not recursive).

    Accepts FreeDLC ``*.fdlc.parquet`` and DeepLabCut ``<video>.parquet``
    alike — the schema decides, not the name. Hidden files are skipped.
    Non-recursive on purpose: a project tree holds backups and derived
    outputs below the source folder that must not be swept up.
    """
    out = []
    d = os.fspath(directory)
    for name in sorted(os.listdir(d)):
        if name.startswith("."):
            continue
        full = os.path.join(d, name)
        if os.path.isfile(full) and is_long_pose_parquet(full):
            out.append(full)
    return out


def pose_parquet_video_name(path: str | os.PathLike) -> str:
    """Video stem a pose parquet belongs to.

    ``<video>.parquet`` -> ``<video>``; ``<video>.fdlc.parquet`` -> ``<video>``.
    """
    stem = os.path.basename(os.fspath(path))
    if stem.lower().endswith(".parquet"):
        stem = stem[: -len(".parquet")]
    if stem.lower().endswith(".fdlc"):
        stem = stem[: -len(".fdlc")]
    return stem


def pose_parquet_bodyparts(path: str | os.PathLike) -> list[str]:
    """Marker names of a pose parquet, in the file's order.

    * long layout with embedded ``bodyparts`` -> those (no data read);
    * long layout otherwise -> unique ``bodypart`` values, first-seen order;
    * wide layout -> bases of the ``<bp>_x/_y/_p`` columns (flat, or the
      innermost level of the IMPORTED_POSE MultiIndex).
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    p = os.fspath(path)
    pf = pq.ParquetFile(p)
    names = pf.schema_arrow.names
    if "bodypart" in names:
        meta = read_pose_metadata(p)
        if meta and meta.get("bodyparts"):
            return [str(b) for b in meta["bodyparts"]]
        col = pq.read_table(p, columns=["bodypart"]).column("bodypart")
        return [str(v) for v in pc.unique(col).to_pylist()]

    try:
        cols = next(pf.iter_batches(batch_size=1)).to_pandas().columns
    except StopIteration:
        cols = names
    labels = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in cols]
    markers: list[str] = []
    for lab in labels:
        for suf in _WIDE_SUFFIXES:
            if lab.endswith(suf):
                bp = lab[: -len(suf)]
                if bp and bp not in markers:
                    markers.append(bp)
                break
    return markers


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def pose_metadata_payload(bodyparts, skeleton, **extra) -> bytes:
    """Serialise the ``b"pose"`` JSON payload (edges as name pairs)."""
    body = {
        "version": POSE_METADATA_VERSION,
        "bodyparts": [str(b) for b in bodyparts],
        "skeleton": [[str(a), str(b)] for a, b in skeleton],
    }
    body.update({k: v for k, v in extra.items() if v is not None})
    return json.dumps(body).encode("utf-8")


def write_parquet_with_pose_metadata(df, path, bodyparts, skeleton, **extra):
    """Write ``df`` to parquet with the ``b"pose"`` metadata attached."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[POSE_METADATA_KEY] = pose_metadata_payload(bodyparts, skeleton, **extra)
    pq.write_table(table.replace_schema_metadata(meta), os.fspath(path))


def embed_pose_metadata(path, skeleton, bodyparts=None, out_path=None) -> str:
    """Add (or replace) the embedded skeleton of an existing pose parquet.

    The data is rewritten unchanged; only the ``b"pose"`` key is set. Other
    metadata (e.g. pandas') is kept.

    :param skeleton: edges as name pairs.
    :param bodyparts: node order; defaults to the file's own markers.
    :param out_path: write elsewhere instead of in place.
    :returns: the path written.
    """
    import pyarrow.parquet as pq

    src = os.fspath(path)
    dst = os.fspath(out_path) if out_path else src
    if bodyparts is None:
        bodyparts = pose_parquet_bodyparts(src)
    edges = [(str(a), str(b)) for a, b in skeleton]
    unknown = sorted({n for e in edges for n in e} - set(bodyparts))
    if unknown:
        raise ValueError(
            f"Skeleton names markers that are not in {os.path.basename(src)}: "
            f"{unknown}. File markers: {list(bodyparts)}."
        )
    table = pq.read_table(src)
    meta = dict(table.schema.metadata or {})
    meta[POSE_METADATA_KEY] = pose_metadata_payload(bodyparts, edges)
    tmp = dst + ".tmp-pose-meta"
    pq.write_table(table.replace_schema_metadata(meta), tmp)
    os.replace(tmp, dst)
    return dst


def skeleton_from_source(source: str | os.PathLike) -> dict | None:
    """Load a skeleton from a project.toml, a DLC config.yaml, a FreeDLC
    ``.fdlc.toml`` sidecar, or another pose parquet."""
    s = os.fspath(source)
    low = s.lower()
    if low.endswith(".parquet"):
        return read_pose_skeleton(s)
    if low.endswith(FDLC_SKELETON_SUFFIX):
        return read_fdlc_skeleton(s)
    if low.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "Reading a DeepLabCut config.yaml needs PyYAML "
                "(pip install pyyaml)."
            ) from exc
        with open(s, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        nodes, edges = _normalize_skeleton(data)
        return {"nodes": nodes, "edges": edges, "source": s} if edges else None
    if low.endswith(".toml"):
        from mufasa.project_layout import read_skeleton
        sk = read_skeleton(s)
        if sk:
            sk["source"] = s
        return sk
    raise ValueError(f"Don't know how to read a skeleton from {s}")


def main(argv=None) -> int:
    """``python -m mufasa.pose_importers.pose_parquet FILE... --skeleton-from SRC``

    Embed a skeleton into existing long-format pose parquets (in place).
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m mufasa.pose_importers.pose_parquet",
        description="Embed a skeleton into long-format pose parquet files.",
    )
    ap.add_argument("files", nargs="+", help="Pose parquet files to update.")
    ap.add_argument(
        "--skeleton-from", required=True,
        help="project.toml, DeepLabCut config.yaml, <stem>.fdlc.toml, or a "
             "parquet that already carries one.",
    )
    args = ap.parse_args(argv)
    sk = skeleton_from_source(args.skeleton_from)
    if not sk or not sk.get("edges"):
        print(f"No skeleton edges found in {args.skeleton_from}.")
        return 1
    rc = 0
    for f in args.files:
        if not is_long_pose_parquet(f):
            print(f"skip {f}: not a long-format pose parquet")
            rc = 1
            continue
        try:
            embed_pose_metadata(f, sk["edges"])
            print(f"embedded {len(sk['edges'])} edges into {f}")
        except ValueError as exc:
            print(f"skip {f}: {exc}")
            rc = 1
    return rc


__all__ = [
    "FDLC_SUFFIX", "FDLC_SKELETON_SUFFIX", "POSE_METADATA_KEY",
    "LONG_COLUMNS", "LONG_REQUIRED_COLUMNS",
    "read_fdlc_skeleton", "read_pose_metadata", "read_embedded_skeleton",
    "read_pose_skeleton", "resolve_skeleton", "is_long_pose_parquet", "find_long_pose_parquets",
    "pose_parquet_video_name", "pose_parquet_bodyparts",
    "pose_metadata_payload", "write_parquet_with_pose_metadata",
    "embed_pose_metadata", "skeleton_from_source", "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
