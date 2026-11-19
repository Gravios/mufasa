"""Run inside Blender's Python (``bpy``): armature + skin mesh -> arrays.

python -m mufasa.bodymodel._blend_export IN.blend OUT.npz [armature] [skin] [decimate]
Patch 122kq. Kept free of other Mufasa imports so it runs in any interpreter with bpy.
"""
import json
import sys

import bpy
import numpy as np


def main(src, out, arm_name="Armature", skin_name="Skin", ratio=0.12):
    bpy.ops.wm.open_mainfile(filepath=src)
    d = bpy.data
    arm = d.objects.get(arm_name) or next((o for o in d.objects if o.type == "ARMATURE"), None)
    if arm is None:
        raise SystemExit("no armature in the file")
    skin = d.objects.get(skin_name)
    if skin is None:   # the mesh with the most vertices deformed by the armature
        cands = [o for o in d.objects if o.type == "MESH" and any(m.type == "ARMATURE" and m.object == arm
                                                                 for m in o.modifiers)]
        if not cands:
            raise SystemExit("no mesh deformed by the armature")
        skin = max(cands, key=lambda o: len(o.data.vertices))
    order, seen = [], set()

    def visit(b):
        if b.name in seen:
            return
        if b.parent:
            visit(b.parent)
        seen.add(b.name)
        order.append(b)
    for b in arm.data.bones:
        visit(b)
    names = [b.name for b in order]
    ix = {n: i for i, n in enumerate(names)}
    A = np.array(arm.matrix_world)
    rest = np.array([A @ np.array(b.matrix_local) for b in order])
    tails = np.array([(A @ np.r_[b.tail_local[:], 1.0])[:3] for b in order])
    parent = np.array([ix[b.parent.name] if b.parent else -1 for b in order])

    # fur colour per face from the first image texture of the skin's material, at the face's UV centroid
    pix, size = None, None
    for slot in skin.material_slots:
        m = slot.material
        if m and m.node_tree:
            for n in m.node_tree.nodes:
                if n.type == "TEX_IMAGE" and n.image is not None and n.image.size[0] > 0:
                    size = tuple(n.image.size)
                    pix = np.array(n.image.pixels[:], np.float32).reshape(size[1], size[0], -1)
                    break
        if pix is not None:
            break
    base = None
    for slot in skin.material_slots:
        m = slot.material
        if m and m.node_tree and "Principled BSDF" in m.node_tree.nodes:
            base = list(m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value)[:3]

    for mod in skin.modifiers:
        mod.show_viewport = False

    def arrays(with_colour):
        dg = bpy.context.evaluated_depsgraph_get()
        e = skin.evaluated_get(dg)
        me = e.to_mesh()
        M = np.array(skin.matrix_world)
        V = np.array([(M @ np.r_[v.co[:], 1.0])[:3] for v in me.vertices])
        F = np.array([p.vertices[:3] for p in me.polygons])
        gi = {g.index: g.name for g in skin.vertex_groups}
        W = np.zeros((len(V), len(names)), np.float32)
        for v in me.vertices:
            for g in v.groups:
                n = gi.get(g.group)
                if n in ix:
                    W[v.index, ix[n]] = g.weight
        W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
        rgb = None
        if with_colour:
            if pix is not None and me.uv_layers:
                uv = me.uv_layers.active.data
                c = np.array([np.mean([uv[li].uv[:] for li in p.loop_indices], 0) for p in me.polygons])
                xs = np.clip((c[:, 0] % 1) * size[0], 0, size[0] - 1).astype(int)
                ys = np.clip((c[:, 1] % 1) * size[1], 0, size[1] - 1).astype(int)
                rgb = pix[ys, xs, :3]
            else:
                rgb = np.tile(np.array(base or [0.3, 0.3, 0.3]), (len(F), 1))
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)        # byte images: pixels are sRGB already
        e.to_mesh_clear()
        return V, F, W, rgb
    V, F, W, rgb = arrays(True)
    dec = skin.modifiers.new("_dec", "DECIMATE")
    dec.ratio = float(ratio)
    Vd, Fd, Wd, _ = arrays(False)
    skin.modifiers.remove(dec)
    meta = {"source": src, "armature": arm.name, "skin": skin.name, "units": "blender units (mm in the "
            "converted mouse)", "unit_scale": bpy.context.scene.unit_settings.scale_length}
    np.savez_compressed(out, names=np.array(names), parent=parent, rest=rest, tails=tails, V=V, F=F, W=W,
                        Vd=Vd, Fd=Fd, Wd=Wd, face_rgb=rgb, meta=json.dumps(meta))
    print(f"rig: {len(names)} bones, {len(V)} vertices ({len(Vd)} for fitting) -> {out}")


if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]     # blender -b --python … -- args
    main(a[0], a[1], *(a[2:4]), *([float(a[4])] if len(a) > 4 else []))
