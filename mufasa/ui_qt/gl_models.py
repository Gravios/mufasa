"""
mufasa.ui_qt.gl_models
======================

Patch 122ky: the 3-D viewer's models drawn on the GPU (OpenGL 3.3).

The meshes of the models (posed rigs, the virtual rodent, loaded meshes) are drawn into an
off-screen framebuffer with a depth buffer and multisampling, and the viewer lays the picture
over its own painting, under the markers. The vertex shader projects exactly as
``Volume3DWidget.project`` does (orbit, orthographic, through the recording camera, and the
mix of the two while an animation moves between them), so models stay on the markers.
Shading: flat, by each face's angle to the view, as the CPU path shades.

Solid models hide what is behind them, faces and models alike (depth buffer); transparent
ones are drawn back to front; wireframes and points are drawn as lines and points.

When OpenGL cannot start (no driver, a remote session), :meth:`GLModelRenderer.render`
returns None and the viewer draws the models with QPainter as before.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtGui import (
    QImage,
    QOffscreenSurface,
    QOpenGLContext,
    QSurfaceFormat,
    QVector3D,
    QVector4D,
)

GL_DEPTH_TEST, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE = 0x0B71, 0x0BE2, 0x0302, 0x0303, 1
GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT = 0x4000, 0x0100
GL_TRIANGLES, GL_LINES, GL_POINTS, GL_FLOAT = 0x0004, 0x0001, 0x0000, 0x1406
GL_PROGRAM_POINT_SIZE, GL_CLIP_DISTANCE0, GL_MULTISAMPLE = 0x8642, 0x3000, 0x809D
GL_LEQUAL, GL_VERSION, GL_RENDERER = 0x0203, 0x1F02, 0x1F01

VERT = """
#version 330 core
in vec3 pos;
uniform int mode;            // 0 perspective, 1 orthographic, 2 recording camera
uniform float blend;         // perspective -> camera mix (animation moves)
uniform vec3 eye, right_, up_, fwd;
uniform float s, so, W, H, dist;
uniform vec4 P0, P1, P2;     // 3x4 camera matrix rows
uniform float k, ox, oy, psign;
uniform float znear, zfar;
uniform float psize;
out vec3 wpos;

vec3 cam(vec3 X) {           // (px, py, depth) through the recording camera
    vec4 h = vec4(X, 1.0);
    float w = dot(P2, h);
    return vec3(ox + k * dot(P0, h) / w, oy + k * dot(P1, h) / w, w * psign);
}

void main() {
    vec3 d = pos - eye;
    float x = dot(d, right_), y = dot(d, up_), z = dot(d, fwd);
    vec3 q;
    if (mode == 2) {
        q = cam(pos);
    } else if (mode == 1) {
        q = vec3(W / 2.0 + so * x, H / 2.0 - so * y, z + 10.0 * dist);
    } else {
        q = vec3(W / 2.0 + s * x / z, H / 2.0 - s * y / z, z);
        if (blend > 0.0) {
            vec3 c = cam(pos);
            q.xy = blend * c.xy + (1.0 - blend) * q.xy;
            if (c.z <= 0.0) q.z = -1.0;
        }
    }
    gl_ClipDistance[0] = q.z - 1e-6;
    float zn = 2.0 * (q.z - znear) / max(zfar - znear, 1e-9) - 1.0;
    gl_Position = vec4(2.0 * q.x / W - 1.0, 1.0 - 2.0 * q.y / H, clamp(zn, -1.0, 1.0), 1.0);
    gl_PointSize = psize;
    wpos = pos;
}
"""

FRAG = """
#version 330 core
in vec3 wpos;
uniform vec4 color;
uniform vec3 fwd;
uniform int shaded;
out vec4 frag;
void main() {
    float sh = 1.0;
    if (shaded == 1) {
        vec3 n = normalize(cross(dFdx(wpos), dFdy(wpos)));
        sh = 0.35 + 0.65 * abs(dot(n, -fwd));
    }
    frag = vec4(color.rgb * sh, color.a);
}
"""


class GLModelRenderer:
    """Off-screen OpenGL drawing of the viewer's model meshes."""

    def __init__(self) -> None:
        self.ok: bool | None = None              # None: not tried yet
        self.error = ""
        self.info = ""
        self.ctx = self.surface = self.prog = self.vao = self.vbo = self.fbo = None
        self._fbo_size = None

    # -- set-up ------------------------------------------------------------ #
    def ensure(self) -> bool:
        if self.ok is not None:
            return self.ok
        try:
            self._init()
            self.ok = True
        except Exception as e:  # noqa: BLE001
            self.ok, self.error = False, str(e) or type(e).__name__
        return self.ok

    def _init(self) -> None:
        from PySide6.QtOpenGL import (
            QOpenGLBuffer,
            QOpenGLShader,
            QOpenGLShaderProgram,
            QOpenGLVertexArrayObject,
        )
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        self.ctx = QOpenGLContext()
        self.ctx.setFormat(fmt)
        if not self.ctx.create():
            raise RuntimeError("no OpenGL context")
        v = self.ctx.format().version()
        if v < (3, 3):
            raise RuntimeError(f"OpenGL {v[0]}.{v[1]} (3.3 needed)")
        self.surface = QOffscreenSurface()
        self.surface.setFormat(self.ctx.format())
        self.surface.create()
        if not self.ctx.makeCurrent(self.surface):
            raise RuntimeError("cannot use the OpenGL context")
        f = self.ctx.functions()
        self.info = f"{f.glGetString(GL_RENDERER)} · OpenGL {f.glGetString(GL_VERSION)}"
        self.prog = QOpenGLShaderProgram()
        if not self.prog.addShaderFromSourceCode(QOpenGLShader.Vertex, VERT):
            raise RuntimeError(self.prog.log())
        if not self.prog.addShaderFromSourceCode(QOpenGLShader.Fragment, FRAG):
            raise RuntimeError(self.prog.log())
        self.prog.bindAttributeLocation("pos", 0)
        if not self.prog.link():
            raise RuntimeError(self.prog.log())
        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo.create()
        self.vbo.setUsagePattern(QOpenGLBuffer.StreamDraw)
        self.ctx.doneCurrent()

    def _framebuffer(self, w: int, h: int):
        from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat
        if self.fbo is None or self._fbo_size != (w, h):
            ff = QOpenGLFramebufferObjectFormat()
            ff.setAttachment(QOpenGLFramebufferObject.Depth)
            ff.setSamples(4)
            self.fbo = QOpenGLFramebufferObject(w, h, ff)
            self._fbo_size = (w, h)
        return self.fbo

    # -- drawing ----------------------------------------------------------- #
    def render(self, w: int, h: int, view: dict, items: list) -> QImage | None:
        """``items``: (V (n, 3) world, F (m, 3), look) per model; ``view``: the viewer's projection
        (see ``Volume3DWidget._gl_view``). A transparent picture of the models, or None."""
        if w < 2 or h < 2 or not self.ensure():
            return None
        try:
            return self._render(int(w), int(h), view, items)
        except Exception as e:  # noqa: BLE001
            self.ok, self.error = False, str(e) or type(e).__name__
            return None

    def _render(self, w, h, view, items) -> QImage:
        self.ctx.makeCurrent(self.surface)
        try:
            f = self.ctx.functions()
            fbo = self._framebuffer(w, h)
            fbo.bind()
            f.glViewport(0, 0, w, h)
            f.glClearColor(0.0, 0.0, 0.0, 0.0)
            f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            f.glEnable(GL_DEPTH_TEST)
            f.glDepthFunc(GL_LEQUAL)
            f.glEnable(GL_CLIP_DISTANCE0)
            f.glEnable(GL_PROGRAM_POINT_SIZE)
            f.glEnable(GL_MULTISAMPLE)
            self.prog.bind()
            self._uniforms(view, w, h, items)
            self.vao.bind()
            self.vbo.bind()
            self.prog.enableAttributeArray(0)
            self.prog.setAttributeBuffer(0, GL_FLOAT, 0, 3, 12)
            # solid first (they fill the depth buffer), then the rest
            order = sorted(items, key=lambda it: it[2].style != "solid")
            for V, F, look in order:
                self._draw(f, V, F, look, view)
            self.vao.release()
            self.prog.release()
            fbo.release()
            img = fbo.toImage(True)
            return img.convertToFormat(QImage.Format_ARGB32_Premultiplied)
        finally:
            self.ctx.doneCurrent()

    def _uniforms(self, view, w, h, items) -> None:
        p = self.prog
        mode = {"perspective": 0, "ortho": 1, "camera": 2}[view["mode"]]
        p.setUniformValue1i(p.uniformLocation("mode"), mode)
        p.setUniformValue1f(p.uniformLocation("blend"), float(view.get("blend", 0.0)))
        for n in ("eye", "right_", "up_", "fwd"):
            p.setUniformValue(p.uniformLocation(n), QVector3D(*map(float, view[n.rstrip("_")])))
        for n in ("s", "so", "dist", "k", "ox", "oy", "psign"):
            p.setUniformValue1f(p.uniformLocation(n), float(view.get(n, 0.0)))
        p.setUniformValue1f(p.uniformLocation("W"), float(w))
        p.setUniformValue1f(p.uniformLocation("H"), float(h))
        P = np.asarray(view.get("P", np.zeros((3, 4))), float)
        for i in range(3):
            p.setUniformValue(p.uniformLocation(f"P{i}"), QVector4D(*map(float, P[i])))
        zn, zf = view["depth_range"]
        p.setUniformValue1f(p.uniformLocation("znear"), float(zn))
        p.setUniformValue1f(p.uniformLocation("zfar"), float(zf))

    def _draw(self, f, V, F, look, view) -> None:
        from PySide6.QtGui import QColor
        p = self.prog
        c = QColor(look.color)
        alpha = max(0.05, min(1.0, float(look.opacity)))
        style = look.style
        if style == "solid":
            alpha = 1.0
        p.setUniformValue(p.uniformLocation("color"), QVector4D(c.redF(), c.greenF(), c.blueF(), alpha))
        p.setUniformValue1i(p.uniformLocation("shaded"), 1 if style in ("solid", "transparent") else 0)
        p.setUniformValue1f(p.uniformLocation("psize"), 2.6)
        V = np.asarray(V, np.float32)
        if style == "points":
            data, prim = V, GL_POINTS
        elif style == "wireframe":
            E = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
            data, prim = V[E].reshape(-1, 3), GL_LINES
        else:
            FF = F
            if style == "transparent":               # back to front
                depth = (V[F].mean(1) - view["eye"]) @ np.asarray(view["fwd"], float)
                FF = F[np.argsort(-depth)]
            data, prim = V[FF].reshape(-1, 3), GL_TRIANGLES
        data = np.ascontiguousarray(data, np.float32)
        if not len(data):
            return
        blend = style != "solid"
        if blend:
            f.glEnable(GL_BLEND)
            # colour by alpha; alpha accumulates once: a premultiplied picture for QPainter
            f.glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA)
            f.glDepthMask(style != "transparent")
        else:
            f.glDisable(GL_BLEND)
            f.glDepthMask(True)
        self.vbo.allocate(data.tobytes(), data.nbytes)
        f.glDrawArrays(prim, 0, len(data))
        f.glDepthMask(True)
        f.glDisable(GL_BLEND)


_shared: GLModelRenderer | None = None


def shared_renderer() -> GLModelRenderer:
    global _shared
    if _shared is None:
        _shared = GLModelRenderer()
    return _shared


__all__ = ["GLModelRenderer", "shared_renderer"]
