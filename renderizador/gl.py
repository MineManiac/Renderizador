#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time
from turtle import color         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    def _get_rgb_from_colors(colors):
        # colors may include emissiveColor as float [0..1] or [0..255]
        rgb = colors.get('emissiveColor') or colors.get('diffuseColor') or [1.0, 1.0, 1.0]
        # normalize to 0..255 ints
        if isinstance(rgb, (list, tuple)) and len(rgb) >= 3:
            r,g,b = rgb[:3]
            # If in 0..1, scale; else assume 0..255
            if (0.0 <= r <= 1.0) and (0.0 <= g <= 1.0) and (0.0 <= b <= 1.0):
                r,g,b = int(r*255), int(g*255), int(b*255)
            else:
                r,g,b = int(r), int(g), int(b)
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))
        else:
            r,g,b = 255,255,255
        return [r,g,b]
    
    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

# ====================== Helpers =================
    # Pilha de Model (identidade na base)
    _model_stack = [[1,0,0,0,
                     0,1,0,0,
                     0,0,1,0,
                     0,0,0,1]]

    # View e Projection (iniciam identidade; viewpoint define)
    _view = [1,0,0,0,
             0,1,0,0,
             0,0,1,0,
             0,0,0,1]

    _proj = [1,0,0,0,
             0,1,0,0,
             0,0,1,0,
             0,0,0,1]
    
    _default_cam_set = False


    @staticmethod
    def _m4_mul(a, b):
        # a,b: 4x4 em row-major (lista de 16)
        c = [0]*16
        for r in range(4):
            for col in range(4):
                c[4*r+col] = (a[4*r+0]*b[0+col] +
                              a[4*r+1]*b[4*1+col] +
                              a[4*r+2]*b[4*2+col] +
                              a[4*r+3]*b[4*3+col])
        return c

    @staticmethod
    def _m4_vec(a, v):  # v = [x,y,z,w]
        return [
            a[0]*v[0] + a[1]*v[1] + a[2]*v[2] + a[3]*v[3],
            a[4]*v[0] + a[5]*v[1] + a[6]*v[2] + a[7]*v[3],
            a[8]*v[0] + a[9]*v[1] + a[10]*v[2] + a[11]*v[3],
            a[12]*v[0] + a[13]*v[1] + a[14]*v[2] + a[15]*v[3],
        ]

    @staticmethod
    def _m4_identity():
        return [1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                0,0,0,1]

    @staticmethod
    def _m4_translation(tx,ty,tz):
        return [1,0,0,tx,
                0,1,0,ty,
                0,0,1,tz,
                0,0,0,1]

    @staticmethod
    def _m4_scale(sx,sy,sz):
        return [sx,0, 0, 0,
                0, sy,0, 0,
                0, 0, sz,0,
                0, 0, 0, 1]

    @staticmethod
    def _m4_rotation_axis_angle(ax, ay, az, t):
        import math
        l = math.sqrt(ax*ax + ay*ay + az*az) or 1.0
        x, y, z = ax/l, ay/l, az/l
        c = math.cos(t); s = math.sin(t); ic = 1.0 - c
        return [x*x*ic + c,     x*y*ic - z*s, x*z*ic + y*s, 0,
                y*x*ic + z*s,   y*y*ic + c,   y*z*ic - x*s, 0,
                z*x*ic - y*s,   z*y*ic + x*s, z*z*ic + c,   0,
                0,              0,            0,            1]

    @staticmethod
    def _m4_inverse_rigid(mat):
        # Inversa de C = T * R (sem escala não-uniforme)
        R = [mat[0],mat[1],mat[2],
             mat[4],mat[5],mat[6],
             mat[8],mat[9],mat[10]]
        t = [mat[3], mat[7], mat[11]]
        Rt = [R[0],R[3],R[6],
              R[1],R[4],R[7],
              R[2],R[5],R[8]]
        tx = -(Rt[0]*t[0] + Rt[1]*t[1] + Rt[2]*t[2])
        ty = -(Rt[3]*t[0] + Rt[4]*t[1] + Rt[5]*t[2])
        tz = -(Rt[6]*t[0] + Rt[7]*t[1] + Rt[8]*t[2])
        return [Rt[0],Rt[1],Rt[2],tx,
                Rt[3],Rt[4],Rt[5],ty,
                Rt[6],Rt[7],Rt[8],tz,
                0,0,0,1]
# --- 3x3 helpers para normais -----------------------------------------------

    @staticmethod
    def _m4_mul_3x3_vec(m, v):
        """
        Multiplica a parte 3x3 de uma matriz por um vetor 3D.
        Aceita m com len == 16 (4x4) ou len == 9 (3x3).
        """
        if len(m) == 16:
            return [
                m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
                m[4]*v[0] + m[5]*v[1] + m[6]*v[2],
                m[8]*v[0] + m[9]*v[1] + m[10]*v[2],
            ]
        elif len(m) == 9:
            return [
                m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
                m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
                m[6]*v[0] + m[7]*v[1] + m[8]*v[2],
            ]
        else:
            raise ValueError("matrix size must be 9 or 16")

    @staticmethod
    def _m3_from_4x4(m4):
        """Extrai a submatriz 3x3 (row-major) de uma 4x4 (len 16)."""
        return [m4[0], m4[1], m4[2],
                m4[4], m4[5], m4[6],
                m4[8], m4[9], m4[10]]

    @staticmethod
    def _m3_transpose(m):
        return [m[0], m[3], m[6],
                m[1], m[4], m[7],
                m[2], m[5], m[8]]

    @staticmethod
    def _m3_inverse(m):
        a,b,c, d,e,f, g,h,i = m
        det = (a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g))
        if det == 0:
            # sem inversa (deixa identidade para não quebrar)
            return [1,0,0, 0,1,0, 0,0,1]
        invdet = 1.0/det
        return [
            (e*i - f*h)*invdet, (c*h - b*i)*invdet, (b*f - c*e)*invdet,
            (f*g - d*i)*invdet, (a*i - c*g)*invdet, (c*d - a*f)*invdet,
            (d*h - e*g)*invdet, (g*b - a*h)*invdet, (a*e - b*d)*invdet
        ]

    @staticmethod
    def _m3_mul_vec(m, v):
        # m: lista 9, v: (x,y,z)
        return [
            m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
            m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
            m[6]*v[0] + m[7]*v[1] + m[8]*v[2],
        ]
    
    @staticmethod
    def _m3_inv_transpose_from_m4(m4):
        # extrai a 3x3 do MV e retorna (MV^{-1})^T (normal matrix)
        a,b,c = m4[0],m4[1],m4[2]
        d,e,f = m4[4],m4[5],m4[6]
        g,h,i = m4[8],m4[9],m4[10]
        det = (a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g))
        if det == 0:
            # devolve identidade pra não quebrar
            return [1,0,0, 0,1,0, 0,0,1]
        invdet = 1.0/det
        inv = [
            (e*i - f*h)*invdet, (c*h - b*i)*invdet, (b*f - c*e)*invdet,
            (f*g - d*i)*invdet, (a*i - c*g)*invdet, (c*d - a*f)*invdet,
            (d*h - e*g)*invdet, (b*g - a*h)*invdet, (a*e - b*d)*invdet
        ]
        # transpose
        return [inv[0],inv[3],inv[6],
                inv[1],inv[4],inv[7],
                inv[2],inv[5],inv[8]]

    @staticmethod
    def _normal_matrix():
        """
        Matriz para transformar normais em view space:
        N = (MV_3x3)^{-T}. Se só houver rotações/translações/escala uniforme,
        funciona também usar direto MV_3x3; aqui uso a forma geral.
        """
        M  = GL._model_stack[-1]
        MV = GL._m4_mul(GL._view, M)
        m3 = GL._m3_from_4x4(MV)
        inv = GL._m3_inverse(m3)
        return GL._m3_transpose(inv)

    @staticmethod
    def _perspective(fovY, aspect, near, far):
        import math
        f = 1.0 / math.tan(fovY * 0.5)
        A = f / aspect
        B = f
        C = (far + near) / (near - far)
        D = (2 * far * near) / (near - far)
        return [A, 0, 0, 0,
                0, B, 0, 0,
                0, 0, C, D,
                0, 0, -1, 0]
    
    @staticmethod
    def _project_point(x, y, z):
        v = [x, y, z, 1.0]
        M   = GL._model_stack[-1]
        VP  = GL._m4_mul(GL._proj, GL._view)
        MVP = GL._m4_mul(VP, M)
        c = GL._m4_vec(MVP, v)          # clip
        if c[3] == 0:
            return None
        invw = 1.0 / c[3]
        x_ndc, y_ndc, z_ndc = c[0]*invw, c[1]*invw, c[2]*invw   # NDC [-1,1]
        sx = (x_ndc * 0.5 + 0.5) * (GL.width  - 1)
        sy = (-y_ndc * 0.5 + 0.5) * (GL.height - 1)
        return (sx, sy, z_ndc, invw)

    @staticmethod
    def _project_point4(x, y, z):
        """Retorna (sx, sy, z_ndc, inv_w, posVS)"""
        v = [x, y, z, 1.0]
        M = GL._model_stack[-1]
        MV = GL._m4_mul(GL._view, M)
        MVP = GL._m4_mul(GL._proj, MV)

        # posição em view space (antes da projeção) para o specular/view dir
        pVS = GL._m4_vec(MV, v)  # [xv,yv,zv,wv] (em VS wv deve ser 1)

        c = GL._m4_vec(MVP, v)
        w = c[3]
        if w == 0:
            return None
        inv_w = 1.0 / w
        x_ndc = c[0] * inv_w
        y_ndc = c[1] * inv_w
        z_ndc = c[2] * inv_w
        sx = (x_ndc * 0.5 + 0.5) * (GL.width  - 1)
        sy = (-y_ndc * 0.5 + 0.5) * (GL.height - 1)
        return (sx, sy, z_ndc, inv_w, [pVS[0], pVS[1], pVS[2]])

    @staticmethod
    def _fill_triangle_screen(p0, p1, p2, rgb):
        # compat: chama o raster novo com cor flat
        GL._raster_triangle(p0, p1, p2, rgb, None, None, base_alpha=1.0)

# --- estado de frame, zbuffer e composição ---
    _ssaa = 2                      # 2x2 amostras por pixel
    _sub_offsets = [(0.25,0.25), (0.75,0.25), (0.25,0.75), (0.75,0.75)]

    _zbuf_sub = None               # [h][w][4] em SSAA 2x2
    _cbuf_sub = None               # [h][w][4][RGBA]
    _cbuf = None                   # resolvido por pixel (RGB)
    _frame_w = None
    _frame_h = None

    @staticmethod
    def _begin_frame():
        """Inicializa/limpa buffers por frame (com SSAA)."""
        need_alloc = (GL._frame_w != GL.width) or (GL._frame_h != GL.height) \
                    or (GL._zbuf_sub is None) or (GL._cbuf_sub is None) or (GL._cbuf is None)
        GL._frame_w, GL._frame_h = GL.width, GL.height
        n = GL._ssaa * GL._ssaa
        if need_alloc:
            GL._zbuf_sub = [[[+1.0 for _ in range(n)] for _ in range(GL.width)] for __ in range(GL.height)]
            GL._cbuf_sub = [[[[0.0,0.0,0.0,0.0] for _ in range(n)] for _ in range(GL.width)] for __ in range(GL.height)]
            GL._cbuf     = [[[0.0,0.0,0.0] for _ in range(GL.width)] for __ in range(GL.height)]
        else:
            for y in range(GL.height):
                for x in range(GL.width):
                    for s in range(n):
                        GL._zbuf_sub[y][x][s] = +1.0
                        GL._cbuf_sub[y][x][s][0] = 0.0
                        GL._cbuf_sub[y][x][s][1] = 0.0
                        GL._cbuf_sub[y][x][s][2] = 0.0
                        GL._cbuf_sub[y][x][s][3] = 0.0
                    GL._cbuf[y][x][0] = GL._cbuf[y][x][1] = GL._cbuf[y][x][2] = 0.0
    
    @staticmethod
    def _resolve_pixel(px, py):
        """Média das subamostras para o framebuffer e escreve na GPU."""
        n = GL._ssaa * GL._ssaa
        acc_r = acc_g = acc_b = 0.0
        for s in range(n):
            acc_r += GL._cbuf_sub[py][px][s][0]
            acc_g += GL._cbuf_sub[py][px][s][1]
            acc_b += GL._cbuf_sub[py][px][s][2]
        r = acc_r / n; g = acc_g / n; b = acc_b / n
        GL._cbuf[py][px][0] = r; GL._cbuf[py][px][1] = g; GL._cbuf[py][px][2] = b
        gpu.GPU.draw_pixel([px, py], gpu.GPU.RGB8, [int(max(0,min(255,r))), int(max(0,min(255,g))), int(max(0,min(255,b)))])
    
    @staticmethod
    def _blend_over(src_rgb, src_a, dst_rgba):
        """Composição 'source over' em 0..255; dst_rgba = [r,g,b,a] (a em [0..1])."""
        a = max(0.0, min(1.0, float(src_a)))
        inv = 1.0 - a
        out_r = src_rgb[0]*a + dst_rgba[0]*inv
        out_g = src_rgb[1]*a + dst_rgba[1]*inv
        out_b = src_rgb[2]*a + dst_rgba[2]*inv
        out_a = a + dst_rgba[3]*(1.0 - a)
        return [out_r, out_g, out_b, out_a]

    @staticmethod
    def _present_pixel(px, py):
        """Atualiza o framebuffer da GPU a partir do _cbuf (resolve SSAA=1)."""
        r, g, b = GL._cbuf[py][px][:3]
        gpu.GPU.draw_pixel([px, py], gpu.GPU.RGB8, [int(max(0, min(255, r))),
                                                    int(max(0, min(255, g))),
                                                    int(max(0, min(255, b)))])

    @staticmethod
    def _raster_triangle(v0, v1, v2, c0, c1, c2, base_alpha=1.0):
        """
        v* = (sx, sy, z_ndc, invw)
        c* = [r,g,b] por vértice (0..255) ou None (usa flat via c0)
        """
        import math
        X0,Y0,Z0,iw0 = v0; X1,Y1,Z1,iw1 = v1; X2,Y2,Z2,iw2 = v2

        def edge(ax, ay, bx, by, cx, cy):
            return (cx - ax)*(by - ay) - (cy - ay)*(bx - ax)

        area = edge(X0, Y0, X1, Y1, X2, Y2)
        if area == 0:
            return
        area_inv = 1.0 / area
        xmin = max(0,             int(math.floor(min(X0, X1, X2))))
        xmax = min(GL.width - 1,  int(math.ceil (max(X0, X1, X2))))
        ymin = max(0,             int(math.floor(min(Y0, Y1, Y2))))
        ymax = min(GL.height - 1, int(math.ceil (max(Y0, Y1, Y2))))

        n = GL._ssaa * GL._ssaa
        offs = GL._sub_offsets if GL._ssaa == 2 else [(0.5,0.5)]
        alpha = max(0.0, min(1.0, base_alpha))

        for py in range(ymin, ymax+1):
            for px in range(xmin, xmax+1):
                for s, (ox, oy) in enumerate(offs):
                    sx = px + ox
                    sy = py + oy

                    w0 = edge(X1, Y1, X2, Y2, sx, sy)
                    w1 = edge(X2, Y2, X0, Y0, sx, sy)
                    w2 = edge(X0, Y0, X1, Y1, sx, sy)
                    inside_pos = (w0 >= 0 and w1 >= 0 and w2 >= 0)
                    inside_neg = (w0 <= 0 and w1 <= 0 and w2 <= 0)
                    if not (inside_pos or inside_neg):
                        continue

                    l0 = w0 * area_inv
                    l1 = w1 * area_inv
                    l2 = w2 * area_inv

                    zndc = l0*Z0 + l1*Z1 + l2*Z2  # NDC: near=-1, far=+1

                    # opaco: z-test por subamostra
                    if alpha >= 0.999:
                        if zndc >= GL._zbuf_sub[py][px][s]:
                            continue
                        GL._zbuf_sub[py][px][s] = zndc

                    # cor: perspectiva-correta se vier c0/c1/c2
                    if (c0 is not None) and (c1 is not None) and (c2 is not None):
                        denom = l0*iw0 + l1*iw1 + l2*iw2
                        if denom != 0:
                            r = (c0[0]*iw0*l0 + c1[0]*iw1*l1 + c2[0]*iw2*l2) / denom
                            g = (c0[1]*iw0*l0 + c1[1]*iw1*l1 + c2[1]*iw2*l2) / denom
                            b = (c0[2]*iw0*l0 + c1[2]*iw1*l1 + c2[2]*iw2*l2) / denom
                            src = [r, g, b]
                        else:
                            src = [0.0, 0.0, 0.0]
                    else:
                        src = c0 if c0 is not None else [255.0, 255.0, 255.0]

                    # escreve (opaco => sobrescreve; transparente => 'over')
                    if alpha >= 0.999:
                        GL._cbuf_sub[py][px][s][0] = src[0]
                        GL._cbuf_sub[py][px][s][1] = src[1]
                        GL._cbuf_sub[py][px][s][2] = src[2]
                        GL._cbuf_sub[py][px][s][3] = 1.0
                    else:
                        GL._cbuf_sub[py][px][s] = GL._blend_over(src, alpha, GL._cbuf_sub[py][px][s])

                # resolve pixel depois de mexer nas amostras
                GL._resolve_pixel(px, py)
    
    # --- tex ---
    @staticmethod
    def _read_tex(tex_handle, u, v):
        import math, numpy as _np
        # wrap
        u = u - math.floor(u)
        v = v - math.floor(v)
        u, v = 1.0 - v, u

        # 1) se for imagem crua (numpy array RGBA)
        if isinstance(tex_handle, _np.ndarray):
            arr = _np.array(tex_handle)
            H, W = arr.shape[0], arr.shape[1]
            x = min(max(int(round(u * (W - 1))), 0), W - 1)
            y = min(max(int(round(v * (H - 1))), 0), H - 1)
            px = arr[y, x]
            return [int(px[0]), int(px[1]), int(px[2])]

        # 2) senão, usa a API da GPU (handle inteiro)
        try:
            return gpu.GPU.read_texture(tex_handle, [u, v])
        except Exception:
            try:
                return gpu.GPU.read_texture(tex_handle, u, v)
            except Exception:
                return [255, 255, 255]
    
    # --- cache de texturas ---
    _tex_cache = {}

    @staticmethod
    def _get_texture_handle(current_texture):
        """
        Retorna algo que represente a textura:
        - se já for int (handle da GPU) -> retorna
        - se for numpy.ndarray (imagem já carregada) -> retorna
        - se for list/tuple/dict/str com caminho -> tenta carregar via GPU.load_texture(...)
        """
        import numpy as _np

        # 1) já é handle ou imagem crua?
        if isinstance(current_texture, int):
            return current_texture
        if isinstance(current_texture, _np.ndarray):
            return current_texture

        # 2) extrai uma 'key' (caminho) de estruturas comuns do X3D
        key = None
        if isinstance(current_texture, (list, tuple)) and current_texture:
            # prioriza um handle inteiro; senão primeira string
            for e in current_texture:
                if isinstance(e, int):
                    return e
            for e in current_texture:
                if isinstance(e, str):
                    key = e
                    break
        elif isinstance(current_texture, dict):
            if isinstance(current_texture.get('handle'), int):
                return current_texture['handle']
            u = current_texture.get('url') or current_texture.get('image')
            if isinstance(u, (list, tuple)) and u:
                key = u[0]
            elif isinstance(u, str):
                key = u
        elif isinstance(current_texture, str):
            key = current_texture

        if not key:
            return None

        # 3) cache e carregamento
        if not hasattr(GL, "_tex_cache"):
            GL._tex_cache = {}
        if key in GL._tex_cache:
            return GL._tex_cache[key]

        try:
            handle = gpu.GPU.load_texture(key)   # pode devolver int ou uma matriz RGBA
        except Exception:
            handle = None
        GL._tex_cache[key] = handle
        return handle

            
    @staticmethod
    def _raster_triangle_tex(v0, v1, v2, uv0, uv1, uv2, tex_handle, mod_colors=None, base_alpha=1.0):
        """
        v*  = (sx, sy, z_ndc, invw)
        uv* = (u, v)
        mod_colors: lista [(r,g,b) 0..255] por vértice OU None (usa só textura).
        """
        import math
        X0,Y0,Z0,iw0 = v0; X1,Y1,Z1,iw1 = v1; X2,Y2,Z2,iw2 = v2
        U0,V0 = uv0; U1,V1 = uv1; U2,V2 = uv2

        def edge(ax, ay, bx, by, cx, cy):
            return (cx - ax)*(by - ay) - (cy - ay)*(bx - ax)

        area = edge(X0, Y0, X1, Y1, X2, Y2)
        if area == 0:
            return
        area_inv = 1.0 / area
        xmin = max(0,             int(math.floor(min(X0, X1, X2))))
        xmax = min(GL.width - 1,  int(math.ceil (max(X0, X1, X2))))
        ymin = max(0,             int(math.floor(min(Y0, Y1, Y2))))
        ymax = min(GL.height - 1, int(math.ceil (max(Y0, Y1, Y2))))

        n = GL._ssaa * GL._ssaa
        offs = GL._sub_offsets if GL._ssaa == 2 else [(0.5,0.5)]
        alpha = max(0.0, min(1.0, base_alpha))

        for py in range(ymin, ymax+1):
            for px in range(xmin, xmax+1):
                for s, (ox, oy) in enumerate(offs):
                    sx = px + ox
                    sy = py + oy

                    w0 = edge(X1, Y1, X2, Y2, sx, sy)
                    w1 = edge(X2, Y2, X0, Y0, sx, sy)
                    w2 = edge(X0, Y0, X1, Y1, sx, sy)
                    inside_pos = (w0 >= 0 and w1 >= 0 and w2 >= 0)
                    inside_neg = (w0 <= 0 and w1 <= 0 and w2 <= 0)
                    if not (inside_pos or inside_neg):
                        continue

                    l0 = w0 * area_inv
                    l1 = w1 * area_inv
                    l2 = w2 * area_inv

                    zndc = l0*Z0 + l1*Z1 + l2*Z2

                    if alpha >= 0.999:
                        if zndc >= GL._zbuf_sub[py][px][s]:
                            continue
                        GL._zbuf_sub[py][px][s] = zndc

                    # UV perspectiva-corretos
                    denom = l0*iw0 + l1*iw1 + l2*iw2
                    if denom == 0:
                        continue
                    up = (U0*iw0*l0 + U1*iw1*l1 + U2*iw2*l2) / denom
                    vp = (V0*iw0*l0 + V1*iw1*l1 + V2*iw2*l2) / denom

                    tex_rgb = GL._read_tex(tex_handle, up, vp)
                    r = float(tex_rgb[0]); g = float(tex_rgb[1]); b = float(tex_rgb[2])

                    # modulação opcional por cor do vértice
                    if mod_colors is not None:
                        # interpola cor de modulação também perspectiva-correta
                        c0,c1,c2 = mod_colors
                        mr = (c0[0]*iw0*l0 + c1[0]*iw1*l1 + c2[0]*iw2*l2) / denom / 255.0
                        mg = (c0[1]*iw0*l0 + c1[1]*iw1*l1 + c2[1]*iw2*l2) / denom / 255.0
                        mb = (c0[2]*iw0*l0 + c1[2]*iw1*l1 + c2[2]*iw2*l2) / denom / 255.0
                        r *= mr; g *= mg; b *= mb

                    src = [r, g, b]
                    if alpha >= 0.999:
                        GL._cbuf_sub[py][px][s][0] = src[0]
                        GL._cbuf_sub[py][px][s][1] = src[1]
                        GL._cbuf_sub[py][px][s][2] = src[2]
                        GL._cbuf_sub[py][px][s][3] = 1.0
                    else:
                        GL._cbuf_sub[py][px][s] = GL._blend_over(src, alpha, GL._cbuf_sub[py][px][s])

                GL._resolve_pixel(px, py)

    @staticmethod
    def _ensure_camera_and_frame():
        """Garante uma câmera/projeção padrão e buffers prontos caso Viewpoint não tenha sido chamado."""
        # Se _proj ainda parece identidade, criamos uma perspective padrão + view ‘olhando’ para a origem
        if not getattr(GL, "_default_cam_set", False):
            # heurística simples para detectar identidade: elementos [0,0]=1,[5]=1,[10]=1,[15]=1
            if (GL._proj[0] == 1 and GL._proj[5] == 1 and GL._proj[10] == 1 and GL._proj[15] == 1):
                # câmera default: eye=(0,0,5), sem rotação, fov ~ 60°
                GL.viewpoint([0.0, 0.0, 5.0], [0.0, 0.0, 1.0, 0.0], math.radians(60.0))
                GL._default_cam_set = True
        # Se buffers ainda não foram criados, inicializa um frame
        if GL._zbuf_sub is None or GL._cbuf is None:
            GL._begin_frame()

    @staticmethod
    def _to01(c):
        """Converte [r,g,b] que podem estar em [0..1] ou [0..255] para floats [0..1]."""
        if c is None:
            return [1.0, 1.0, 1.0]
        r, g, b = (float(c[0]), float(c[1]), float(c[2]))
        if max(r, g, b) > 1.0:
            r /= 255.0; g /= 255.0; b /= 255.0
        # clamp
        r = 0.0 if r < 0.0 else (1.0 if r > 1.0 else r)
        g = 0.0 if g < 0.0 else (1.0 if g > 1.0 else g)
        b = 0.0 if b < 0.0 else (1.0 if b > 1.0 else b)
        return [r, g, b]
    
# ====================== iluminação =================
    _headlight = False
    _dir_lights = []   # cada item: { 'ambient': float, 'color': (r,g,b), 'intensity': float, 'dir': (x,y,z) } em VIEW space
    _point_lights = [] # idem com 'pos': (x,y,z)
    _ambient_global = 0.0  # ambient mínimo para não ficar tudo preto

    @staticmethod
    def _normalize3(v):
        x,y,z = v[0], v[1], v[2]
        n = math.sqrt(x*x + y*y + z*z)
        if n == 0: return [0.0,0.0,1.0]
        return [x/n, y/n, z/n]

    @staticmethod
    def _dot(a,b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    @staticmethod
    def _sub(a,b): return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

    @staticmethod
    def _cross(a,b):
        return [a[1]*b[2]-a[2]*b[1],
                a[2]*b[0]-a[0]*b[2],
                a[0]*b[1]-a[1]*b[0]]

    @staticmethod
    def _inverse_transpose_3x3_of(m4):
        # m4 é 4x4 row-major; pega 3x3, inverte e transpõe (normal matrix)
        a = [[m4[0],m4[1],m4[2]],
            [m4[4],m4[5],m4[6]],
            [m4[8],m4[9],m4[10]]]
        # inversa 3x3
        det = (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1]) -
            a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0]) +
            a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))
        if abs(det) < 1e-8:
            return [1,0,0, 0,1,0, 0,0,1]
        invdet = 1.0/det
        m00 = (a[1][1]*a[2][2]-a[1][2]*a[2][1])*invdet
        m01 = (a[0][2]*a[2][1]-a[0][1]*a[2][2])*invdet
        m02 = (a[0][1]*a[1][2]-a[0][2]*a[1][1])*invdet
        m10 = (a[1][2]*a[2][0]-a[1][0]*a[2][2])*invdet
        m11 = (a[0][0]*a[2][2]-a[0][2]*a[2][0])*invdet
        m12 = (a[0][2]*a[1][0]-a[0][0]*a[1][2])*invdet
        m20 = (a[1][0]*a[2][1]-a[1][1]*a[2][0])*invdet
        m21 = (a[0][1]*a[2][0]-a[0][0]*a[2][1])*invdet
        m22 = (a[0][0]*a[1][1]-a[0][1]*a[1][0])*invdet
        # transposta da inversa
        return [m00, m10, m20,
                m01, m11, m21,
                m02, m12, m22]

    @staticmethod
    def _project_point_full(x, y, z):
        # Retorna: (sx, sy, z_ndc, invw, [vx, vy, vz] em view space)
        v = [x, y, z, 1.0]

        M  = GL._model_stack[-1]
        MV = GL._m4_mul(GL._view, M)
        MVP = GL._m4_mul(GL._proj, MV)

        c = GL._m4_vec(MVP, v)   # clip
        if c[3] == 0:
            return None
        invw = 1.0 / c[3]
        x_ndc = c[0] * invw
        y_ndc = c[1] * invw
        z_ndc = c[2] * invw

        sx = (x_ndc * 0.5 + 0.5) * (GL.width  - 1)
        sy = (-y_ndc * 0.5 + 0.5) * (GL.height - 1)

        vv = GL._m4_vec(MV, v)  # posição em view space
        posVS = [vv[0], vv[1], vv[2]]

        return (sx, sy, z_ndc, invw, posVS)

    @staticmethod
    def _shade_vertex(posVS, normalVS, material):
        def norm(v):
            l = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) or 1.0
            return [v[0]/l, v[1]/l, v[2]/l]

        # material
        def rgb01(name, default):
            v = material.get(name, default)
            return v if (0<=v[0]<=1 and 0<=v[1]<=1 and 0<=v[2]<=1) else [v[0]/255.0, v[1]/255.0, v[2]/255.0]
        Kd = rgb01("diffuseColor",  [1.0,1.0,1.0])
        Ks = rgb01("specularColor", [0.0,0.0,0.0])
        Ke = rgb01("emissiveColor", [0.0,0.0,0.0])
        shin = float(material.get("shininess", 0.0))  # X3D: 0..1
        shin_exp = max(1.0, shin*128.0)

        # luz em VIEW
        if GL._lights.get("headlight", False):
            Lvs = [0.0, 0.0, -1.0]                       # direção da luz em VS
        else:
            V = GL._view
            R3 = [V[0],V[1],V[2], V[4],V[5],V[6], V[8],V[9],V[10]]  # 3x3 da view
            dW = GL._lights.get("dir_world", [0.0,0.0,-1.0])
            Lvs = GL._m3_mul_vec(R3, dW)
        L = norm([-Lvs[0], -Lvs[1], -Lvs[2]])           # X3D: direção “para onde viaja” -> usa -dir
        Vdir = norm([-posVS[0], -posVS[1], -posVS[2]])  # olho em (0,0,0) em VS
        N = norm(normalVS)
        if N[0]*Vdir[0] + N[1]*Vdir[1] + N[2]*Vdir[2] < 0.0:
            N = [-N[0], -N[1], -N[2]]

        Lcol = GL._lights.get("color", [1.0,1.0,1.0])
        Lint = GL._lights.get("intensity", 1.0)
        Lamb = max(0.0, float(GL._lights.get("ambientIntensity", 0.0)))  # sem piso global

        ndotl = max(0.0, N[0]*L[0] + N[1]*L[1] + N[2]*L[2])
        # Blinn-Phong
        H = norm([L[0]+Vdir[0], L[1]+Vdir[1], L[2]+Vdir[2]])
        ndoth = max(0.0, N[0]*H[0] + N[1]*H[1] + N[2]*H[2])
        spec_s = (ndoth ** shin_exp)

        amb  = [Lamb*Kd[0], Lamb*Kd[1], Lamb*Kd[2]]
        diff = [Lint*ndotl*Kd[0]*Lcol[0],
                Lint*ndotl*Kd[1]*Lcol[1],
                Lint*ndotl*Kd[2]*Lcol[2]]
        spec = [Lint*spec_s*Ks[0]*Lcol[0],
                Lint*spec_s*Ks[1]*Lcol[1],
                Lint*spec_s*Ks[2]*Lcol[2]]
        rgb = [Ke[0]+amb[0]+diff[0]+spec[0],
            Ke[1]+amb[1]+diff[1]+spec[1],
            Ke[2]+amb[2]+diff[2]+spec[2]]
        return [int(max(0,min(255, rgb[0]*255))),
                int(max(0,min(255, rgb[1]*255))),
                int(max(0,min(255, rgb[2]*255)))]
    
# ====================== 1.1: RASTER 2D =================
    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D (Projeto 1.1)."""
        # Espera-se pares [x0,y0, x1,y1, ...]
        rgb = GL._get_rgb_from_colors(colors)
        # Desenha cada ponto
        n = len(point)//2
        for i in range(n):
            x = int(round(point[2*i + 0]))
            y = int(round(point[2*i + 1]))
            if 0 <= x < GL.width and 0 <= y < GL.height:          # <-- bounds check
                gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, rgb)

        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Renderiza Polyline2D conectando pontos consecutivos (Projeto 1.1)."""
        rgb = GL._get_rgb_from_colors(colors)
        # Helper Bresenham
        def draw_line(x0,y0,x1,y1):
            x0,y0,x1,y1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
            dx, dy = abs(x1-x0), -abs(y1-y0)
            sx, sy = (1 if x0<x1 else -1), (1 if y0<y1 else -1)
            err = dx + dy
            while True:
                if 0 <= x0 < GL.width and 0 <= y0 < GL.height:        # <-- bounds check
                    gpu.GPU.draw_pixel([x0, y0], gpu.GPU.RGB8, rgb)
                if x0==x1 and y0==y1: break
                e2 = 2*err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy
        # Percorre pontos consecutivos
        n = len(lineSegments)//2
        if n < 2:
            return
        xs = [lineSegments[2*i] for i in range(n)]
        ys = [lineSegments[2*i+1] for i in range(n)]
        for i in range(n-1):
            draw_line(xs[i], ys[i], xs[i+1], ys[i+1])


    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).

        print("Circle2D : radius = {0}".format(radius)) # imprime no terminal
        print("Circle2D : colors = {0}".format(colors)) # imprime no terminal as cores
        
        # Exemplo:
        pos_x = GL.width//2
        pos_y = GL.height//2
        gpu.GPU.draw_pixel([pos_x, pos_y], gpu.GPU.RGB8, [255, 0, 255])  # altera pixel (u, v, tipo, r, g, b)
        # cuidado com as cores, o X3D especifica de (0,1) e o Framebuffer de (0,255)


    @staticmethod    
    def triangleSet2D(vertices, colors):
        """Preenche triângulos 2D (Projeto 1.1)."""
        import math
        # usa emissiveColor; se vier em [0..1], converte pra [0..255]
        rgb = colors.get('emissiveColor') or colors.get('diffuseColor') or [1.0, 1.0, 1.0]
        if all(0.0 <= c <= 1.0 for c in rgb[:3]):
            rgb = [int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)]
        else:
            rgb = [int(rgb[0]), int(rgb[1]), int(rgb[2])]
        rgb = [max(0, min(255, v)) for v in rgb]

        def edge(ax, ay, bx, by, cx, cy):
            # função de aresta (orientada)
            return (cy - ay) * (bx - ax) - (cx - ax) * (by - ay)

        assert len(vertices) % 6 == 0, "TriangleSet2D: número de valores deve ser múltiplo de 6"
        for i in range(0, len(vertices), 6):
            x0, y0 = vertices[i+0], vertices[i+1]
            x1, y1 = vertices[i+2], vertices[i+3]
            x2, y2 = vertices[i+4], vertices[i+5]

            # área (det) — pula triângulos degenerados
            area = edge(x0, y0, x1, y1, x2, y2)
            if area == 0:
                continue

            # bounding box (clamp às dimensões da tela)
            xmin = max(0, int(math.floor(min(x0, x1, x2))))
            xmax = min(GL.width - 1,  int(math.ceil (max(x0, x1, x2))))
            ymin = max(0, int(math.floor(min(y0, y1, y2))))
            ymax = min(GL.height - 1, int(math.ceil (max(y0, y1, y2))))

            # varre a caixa e testa baricêntricas por sinal (aceita CW ou CCW)
            for y in range(ymin, ymax + 1):
                for x in range(xmin, xmax + 1):
                    w0 = edge(x0, y0, x1, y1, x, y)
                    w1 = edge(x1, y1, x2, y2, x, y)
                    w2 = edge(x2, y2, x0, y0, x, y)
                    # ponto está dentro se os três tiverem o mesmo sinal (>=0) ou (<=0)
                    inside_pos = (w0 >= 0) and (w1 >= 0) and (w2 >= 0)
                    inside_neg = (w0 <= 0) and (w1 <= 0) and (w2 <= 0)
                    if inside_pos or inside_neg:
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, rgb)


# ====================== 1.2: RASTER 3D =================
    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.

        GL._ensure_camera_and_frame()
        assert len(point) % 9 == 0, "TriangleSet espera múltiplos de 9"
        MV = GL._m4_mul(GL._view, GL._model_stack[-1])
        Nmat = GL._m3_inv_transpose_from_m4(MV)

        def area2(p0, p1, p2):
            return (p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])

        for i in range(0, len(point), 9):
            a = (point[i+0], point[i+1], point[i+2])
            b = (point[i+3], point[i+4], point[i+5])
            c = (point[i+6], point[i+7], point[i+8])

            p0 = GL._project_point4(*a)
            p1 = GL._project_point4(*b)
            p2 = GL._project_point4(*c)
            if not (p0 and p1 and p2): 
                continue

            material = colors or {}
            em = material.get("emissiveColor", [0.0, 0.0, 0.0])
            is_emissive = any(c > 0 for c in (em[:3] if isinstance(em, (list, tuple)) else [0,0,0]))

            # só fazemos culling quando NÃO for emissivo
            if (not is_emissive) and area2(p0, p1, p2) >= 0:
                continue

            # normal por face no WORLD → transforma para VIEW pela Nmat
            # (compute em world: (b-a)x(c-a))
            ax,ay,az = a
            bx,by,bz = b
            cx,cy,cz = c
            U = [bx-ax, by-ay, bz-az]
            Vv= [cx-ax, cy-ay, cz-az]
            Nw = [ U[1]*Vv[2]-U[2]*Vv[1],
                U[2]*Vv[0]-U[0]*Vv[2],
                U[0]*Vv[1]-U[1]*Vv[0] ]
            Nv = GL._m3_mul_vec(Nmat, Nw)

            # cores por vértice (flat com a mesma normal)
            c0 = GL._shade_vertex(p0[4], Nv, colors or {})
            c1 = GL._shade_vertex(p1[4], Nv, colors or {})
            c2 = GL._shade_vertex(p2[4], Nv, colors or {})

            # passa 4 componentes (sx,sy,z_ndc,inv_w)
            GL._raster_triangle( (p0[0],p0[1],p0[2],p0[3]),
                                (p1[0],p1[1],p1[2],p1[3]),
                                (p2[0],p2[1],p2[2],p2[3]),
                                c0,c1,c2,
                                base_alpha=1.0)



    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.

        eye = position or [0.0,0.0,5.0]
        if orientation: ax,ay,az,ang = orientation
        else:           ax,ay,az,ang = 0.0,0.0,1.0,0.0
        T = GL._m4_translation(eye[0], eye[1], eye[2])
        R = GL._m4_rotation_axis_angle(ax, ay, az, ang)
        C = GL._m4_mul(T, R)                 # câmera no mundo
        GL._view = GL._m4_inverse_rigid(C)   # view = C^{-1}
        aspect = GL.width / GL.height if GL.height else 1.0
        GL._proj = GL._perspective(fieldOfView, aspect, GL.near, GL.far)

        # <<< limpar buffers por frame >>>
        GL._begin_frame()
        # limpar estado de luzes a cada frame
        GL._dir_lights = []
        GL._point_lights = []

        # luz padrão caso a cena não informe nenhuma nesta iteração
        GL._lights = getattr(GL, "_lights", {})
        if "headlight" not in GL._lights and "dir_world" not in GL._lights:
            GL._lights["headlight"] = True            # liga o headlight
            GL._lights["ambientIntensity"] = 0.20     # um pouco de ambiente
            GL._lights["color"] = [1.0, 1.0, 1.0]
            GL._lights["intensity"] = 1.0
    

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # ESSES NÃO SÃO OS VALORES DE QUATÉRNIOS AS CONTAS AINDA PRECISAM SER FEITAS.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas. 
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        parent = GL._model_stack[-1]
        tx,ty,tz = translation or [0.0,0.0,0.0]
        sx,sy,sz = scale or [1.0,1.0,1.0]
        if rotation:
            rx,ry,rz,ang = rotation
            R = GL._m4_rotation_axis_angle(rx,ry,rz,ang)
        else:
            R = GL._m4_identity()
        T = GL._m4_translation(tx,ty,tz)
        S = GL._m4_scale(sx,sy,sz)
        local = GL._m4_mul(GL._m4_mul(T,R), S)   # T · R · S
        GL._model_stack.append(GL._m4_mul(parent, local))

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        if len(GL._model_stack) > 1:
            GL._model_stack.pop()

    
    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        GL._ensure_camera_and_frame()
        rgb = GL._get_rgb_from_colors(colors)
        alpha = 1.0 - float(colors.get('transparency', 0.0)) if isinstance(colors, dict) else 1.0
        verts = [(point[i], point[i+1], point[i+2]) for i in range(0, len(point), 3)]
        cursor = 0
        for count in stripCount:
            count = int(count)
            if count < 3:
                cursor += count
                continue
            for i in range(cursor+2, cursor+count):
                a = verts[i-2]; b = verts[i-1]; c = verts[i]
                # alterna winding para manter orientação consistente
                if (i - cursor) % 2 == 0:
                    a, b = b, a
                p0 = GL._project_point(*a)
                p1 = GL._project_point(*b)
                p2 = GL._project_point(*c)
                if p0 and p1 and p2:
                    GL._raster_triangle(p0, p1, p2, rgb, None, None, base_alpha=alpha)
            cursor += count



# ====================== 1.3: STRIPS / INDEXADOS / FACESET ===============
    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        GL._ensure_camera_and_frame()
        rgb = GL._get_rgb_from_colors(colors)
        alpha = 1.0 - float(colors.get('transparency', 0.0)) if isinstance(colors, dict) else 1.0
        verts = [(point[i], point[i+1], point[i+2]) for i in range(0, len(point), 3)]

        strip = []
        def flush(strip):
            if len(strip) < 3:
                return
            for i in range(2, len(strip)):
                a = verts[strip[i-2]]
                b = verts[strip[i-1]]
                c = verts[strip[i]]
                if (i % 2) == 0:
                    a, b = b, a
                p0 = GL._project_point(*a)
                p1 = GL._project_point(*b)
                p2 = GL._project_point(*c)
                if p0 and p1 and p2:
                    GL._raster_triangle(p0, p1, p2, rgb, None, None, base_alpha=alpha)

        for idx in index:
            if idx == -1:
                flush(strip); strip = []
            else:
                strip.append(int(idx))
        flush(strip)  # última tira


    
    @staticmethod
    def indexedFaceSet(coord=None, coordIndex=None, colorPerVertex=True, color=None, colorIndex=None,
                    texCoord=None, texCoordIndex=None, colors=None, current_texture=None):

        GL._ensure_camera_and_frame()
        verts = []
        if coord:
            verts = [(coord[i], coord[i+1], coord[i+2]) for i in range(0, len(coord), 3)]

        # (opcional) paleta de cores por vértice, mas usaremos principalmente material/luz
        # deixo como está se você já usa, não é obrigatório para difusos.

        MV = GL._m4_mul(GL._view, GL._model_stack[-1])
        Nmat = GL._m3_inv_transpose_from_m4(MV)

        def area2(p0, p1, p2):
            return (p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])

        face = []
        for idx in (coordIndex or []):
            if idx == -1:
                if len(face) >= 3:
                    aI = face[0]
                    for k in range(2, len(face)):
                        bI = face[k-1]; cI = face[k]
                        A = verts[aI]; B = verts[bI]; C = verts[cI]

                        p0 = GL._project_point4(*A)
                        p1 = GL._project_point4(*B)
                        p2 = GL._project_point4(*C)
                        if not (p0 and p1 and p2):
                            continue

                        if area2(p0,p1,p2) >= 0:
                            continue

                        # normal de face em WORLD → VIEW
                        U = [B[0]-A[0], B[1]-A[1], B[2]-A[2]]
                        Vv= [C[0]-A[0], C[1]-A[1], C[2]-A[2]]
                        Nw = [ U[1]*Vv[2]-U[2]*Vv[1],
                            U[2]*Vv[0]-U[0]*Vv[2],
                            U[0]*Vv[1]-U[1]*Vv[0] ]
                        Nv = GL._m3_mul_vec(Nmat, Nw)

                        c0 = GL._shade_vertex(p0[4], Nv, colors or {})
                        c1 = GL._shade_vertex(p1[4], Nv, colors or {})
                        c2 = GL._shade_vertex(p2[4], Nv, colors or {})

                        GL._raster_triangle( (p0[0],p0[1],p0[2],p0[3]),
                                            (p1[0],p1[1],p1[2],p1[3]),
                                            (p2[0],p2[1],p2[2],p2[3]),
                                            c0,c1,c2,
                                            base_alpha=1.0)
                face = []
            else:
                face.append(int(idx))

        if len(face) >= 3:
            aI = face[0]
            for k in range(2, len(face)):
                bI = face[k-1]; cI = face[k]
                A = verts[aI]; B = verts[bI]; C = verts[cI]
                p0 = GL._project_point4(*A)
                p1 = GL._project_point4(*B)
                p2 = GL._project_point4(*C)
                if not (p0 and p1 and p2): 
                    continue
                if area2(p0, p1, p2) >= 0:
                    continue

                U = [B[0]-A[0], B[1]-A[1], B[2]-A[2]]
                Vv= [C[0]-A[0], C[1]-A[1], C[2]-A[2]]
                Nw = [ U[1]*Vv[2]-U[2]*Vv[1],
                    U[2]*Vv[0]-U[0]*Vv[2],
                    U[0]*Vv[1]-U[1]*Vv[0] ]
                Nv = GL._m3_mul_vec(Nmat, Nw)

                c0 = GL._shade_vertex(p0[4], Nv, colors or {})
                c1 = GL._shade_vertex(p1[4], Nv, colors or {})
                c2 = GL._shade_vertex(p2[4], Nv, colors or {})

                GL._raster_triangle( (p0[0],p0[1],p0[2],p0[3]),
                                    (p1[0],p1[1],p1[2],p1[3]),
                                    (p2[0],p2[1],p2[2],p2[3]),
                                    c0,c1,c2,
                                    base_alpha=1.0)




    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        @staticmethod
        def sphere(radius, colors):
            """
            Renderiza uma esfera centrada na origem usando latitude/longitude.
            Usa shading por vértice com a sua _shade_vertex (difuso+spec).
            """
            # qualidade (aumente/diminua conforme o FPS desejado)
            stacks = 10   # linhas de latitude (>= 2)
            slices = 16   # colunas de longitude (>= 3)

            r = float(radius)
            if r <= 0:
                return

            import math
            # Matrizes para transformar normais ao espaço de visualização
            MV   = GL._m4_mul(GL._view, GL._model_stack[-1])
            Nmat = GL._m3_inv_transpose_from_m4(MV)

            def area2(p0, p1, p2):
                return (p1[0]-p0[0])*(p2[1]-p0[1]) - (p1[1]-p0[1])*(p2[0]-p0[0])

            # Gera “anéis” (stacks) de latitude de -pi/2..+pi/2
            for i in range(stacks):
                # phi: latitude inferior e superior
                phi0 = math.pi * (-0.5 + i    / stacks)
                phi1 = math.pi * (-0.5 + (i+1)/ stacks)
                y0   = r * math.sin(phi0)
                y1   = r * math.sin(phi1)
                rc0  = r * math.cos(phi0)
                rc1  = r * math.cos(phi1)

                for j in range(slices):
                    # theta: longitude
                    theta0 = 2.0*math.pi * ( j    / slices)
                    theta1 = 2.0*math.pi * ((j+1) / slices)

                    x00 = rc0*math.cos(theta0); z00 = rc0*math.sin(theta0)
                    x01 = rc0*math.cos(theta1); z01 = rc0*math.sin(theta1)
                    x10 = rc1*math.cos(theta0); z10 = rc1*math.sin(theta0)
                    x11 = rc1*math.cos(theta1); z11 = rc1*math.sin(theta1)

                    A = (x00, y0, z00)
                    B = (x10, y1, z10)
                    C = (x11, y1, z11)
                    D = (x01, y0, z01)

                    # Projetamos cada vértice e pegamos posVS para o shader
                    pA = GL._project_point4(*A)
                    pB = GL._project_point4(*B)
                    pC = GL._project_point4(*C)
                    pD = GL._project_point4(*D)
                    if not (pA and pB and pC and pD):
                        continue

                    # Normais no espaço do objeto ≈ posição normalizada (esfera)
                    # Depois levamos para VIEW space com a normal matrix.
                    def n_view(x,y,z):
                        # normal objeto
                        ln = math.sqrt(x*x + y*y + z*z) or 1.0
                        n  = [x/ln, y/ln, z/ln]
                        return GL._m3_mul_vec(Nmat, n)

                    nA = n_view(*A)
                    nB = n_view(*B)
                    nC = n_view(*C)
                    nD = n_view(*D)

                    # Cores por vértice via shader de iluminação
                    cA = GL._shade_vertex(pA[4], nA, colors or {})
                    cB = GL._shade_vertex(pB[4], nB, colors or {})
                    cC = GL._shade_vertex(pC[4], nC, colors or {})
                    cD = GL._shade_vertex(pD[4], nD, colors or {})

                    # Dois triângulos por quad (A-B-C) e (A-C-D)
                    # Faz culling pela orientação em tela para evitar excesso de fill
                    if area2(pA, pB, pC) < 0:
                        GL._raster_triangle(
                            (pA[0],pA[1],pA[2],pA[3]),
                            (pB[0],pB[1],pB[2],pB[3]),
                            (pC[0],pC[1],pC[2],pC[3]),
                            cA, cB, cC, base_alpha=1.0
                        )
                    if area2(pA, pC, pD) < 0:
                        GL._raster_triangle(
                            (pA[0],pA[1],pA[2],pA[3]),
                            (pC[0],pC[1],pC[2],pC[3]),
                            (pD[0],pD[1],pD[2],pD[3]),
                            cA, cC, cD, base_alpha=1.0
                        )

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    
    # estado simples de luz (já existia algo? mantenha, só garanta essas chaves)
    _lights = {
        "headlight": True,
        "dir": [0.0, 0.0, -1.0],  # em view space quando headlight
        "color": [1.0, 1.0, 1.0],
        "intensity": 1.0,
        "ambientIntensity": 0.15,  # um mínimo para não ficar tudo preto
    }

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        GL._lights["headlight"] = bool(headlight)

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # Armazena a luz direcional definida na cena (em WORLD).
        # Convertida para view space quando usada no shader.
        Lc = GL._to01(color)
        GL._lights.setdefault("directionals", []).append({
            "dir_world": [float(direction[0]), float(direction[1]), float(direction[2])],
            "color": Lc,
            "intensity": float(intensity),
            "ambient": float(ambientIntensity),
        })

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # transforma posição pelo ModelView atual
        r, g, b = GL._to01(color)
        # posição em view space
        # usa MV( location,1 )
        M = GL._m4_mul(GL._view, GL._model_stack[-1])
        lx,ly,lz,_ = GL._m4_vec(M, [location[0], location[1], location[2], 1.0])
        GL._point_lights.append({
            'ambient': float(ambientIntensity),
            'color':   (r,g,b),
            'intensity': float(intensity),
            'pos': (lx,ly,lz)
        })

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))


    @staticmethod
    def sphere(radius, colors):
        """Desenha uma esfera por tesselação UV e envia para o indexedFaceSet."""
        r = float(radius)
        stacks = 16   # aumente para 24/32 se quiser mais suavidade
        slices = 24

        coord = []        # [x0,y0,z0, x1,y1,z1, ...]
        coordIndex = []   # triângulos com -1 separando as faces

        # Geração de vértices (latitude = stacks, longitude = slices)
        for i in range(stacks + 1):
            v = i / float(stacks)
            phi = v * math.pi                      # 0..π
            y = r * math.cos(phi)
            sinp = math.sin(phi)
            for j in range(slices + 1):
                u = j / float(slices)
                theta = u * 2.0 * math.pi          # 0..2π
                x = r * sinp * math.cos(theta)
                z = r * sinp * math.sin(theta)
                coord.extend([x, y, z])

        # Helper para index linear no grid
        def idx(ii, jj):
            return ii * (slices + 1) + jj

        # Triangulação em duas faces por quad (a,b,c) e (a,c,d)
        for i in range(stacks):
            for j in range(slices):
                a = idx(i,     j)
                b = idx(i,     j+1)
                c = idx(i+1,   j+1)
                d = idx(i+1,   j)
                coordIndex.extend([a, b, c, -1,  a, c, d, -1])

        # Envia pro seu pipeline (iluminação/transformações já tratadas no indexedFaceSet)
        GL.indexedFaceSet(coord=coord, coordIndex=coordIndex, colors=colors)

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        epoch = time.time()
        t = (epoch % max(1e-6, cycleInterval)) / max(1e-6, cycleInterval)
        return t

    @staticmethod
    def splinePositionInterpolator(key=None, keyValue=None, set_fraction=0.0, closed=False, **_):
        """
        SplinePositionInterpolator (X3D):
        - key:    [k0..kn]  com 0..1
        - keyValue: [x,y,z] para cada key (flatten)
        - set_fraction: f em [0..1]
        Retorna [x,y,z] interpolado. Usa Catmull-Rom; se não der, usa linear.
        """
        if not key or not keyValue:
            return [0.0, 0.0, 0.0]

        # monta lista de pontos 3D
        P = [(keyValue[i], keyValue[i+1], keyValue[i+2]) for i in range(0, len(keyValue), 3)]
        K = list(key)

        # clamp f
        f = max(0.0, min(1.0, float(set_fraction)))

        # caso trivial: 1 ponto
        if len(P) == 1:
            return list(P[0])

        # acha segmento i tal que K[i] <= f <= K[i+1]
        # se f antes do primeiro ou depois do último, clampa nas extremidades
        if f <= K[0]:
            i, u = 0, 0.0
        elif f >= K[-1]:
            i, u = len(K) - 2, 1.0
        else:
            i = max(0, min(len(K) - 2, next(j for j in range(len(K)-1) if K[j] <= f <= K[j+1])))
            denom = (K[i+1] - K[i]) or 1e-8
            u = (f - K[i]) / denom

        # pega P_(i-1), P_i, P_(i+1), P_(i+2) com bordas tratadas
        def at(idx):
            if closed:
                return P[idx % len(P)]
            return P[max(0, min(len(P)-1, idx))]

        P0 = at(i-1); P1 = at(i); P2 = at(i+1); P3 = at(i+2)

        # Catmull-Rom
        def cr(p0, p1, p2, p3, t):
            t2 = t*t; t3 = t2*t
            return (
                0.5 * ( (2*p1) + (-p0 + p2)*t + (2*p0 - 5*p1 + 4*p2 - p3)*t2 + (-p0 + 3*p1 - 3*p2 + p3)*t3 )
            )

        # se tiver poucos pontos, cai no linear
        if len(P) < 4 and not closed:
            a = P[i]; b = P[i+1]
            return [a[0]*(1-u) + b[0]*u, a[1]*(1-u) + b[1]*u, a[2]*(1-u) + b[2]*u]

        x = cr(P0[0], P1[0], P2[0], P3[0], u)
        y = cr(P0[1], P1[1], P2[1], P3[1], u)
        z = cr(P0[2], P1[2], P2[2], P3[2], u)
        return [x, y, z]

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        """Interpola rotações por SLERP (quaternions). keyValue vem em eixos-ângulo."""
        if not key or not keyValue:
            return [0.0, 0.0, 1.0, 0.0]

        R = [(keyValue[i], keyValue[i+1], keyValue[i+2], keyValue[i+3])
            for i in range(0, len(keyValue), 4)]
        K = list(key)
        f = max(0.0, min(1.0, float(set_fraction)))

        if len(R) == 1:
            return list(R[0])

        # acha segmento
        if f <= K[0]:
            i, u = 0, 0.0
        elif f >= K[-1]:
            i, u = len(K)-2, 1.0
        else:
            i = max(0, min(len(K)-2, next(j for j in range(len(K)-1) if K[j] <= f <= K[j+1])))
            denom = (K[i+1] - K[i]) or 1e-8
            u = (f - K[i]) / denom

        import math

        def axis_angle_to_q(ax, ay, az, ang):
            n = math.sqrt(ax*ax + ay*ay + az*az) or 1.0
            ax, ay, az = ax/n, ay/n, az/n
            s = math.sin(ang*0.5)
            return (math.cos(ang*0.5), ax*s, ay*s, az*s)  # (w, x, y, z)

        def q_to_axis_angle(q):
            w, x, y, z = q
            # normaliza
            n = math.sqrt(w*w + x*x + y*y + z*z) or 1.0
            w, x, y, z = w/n, x/n, y/n, z/n
            ang = 2*math.acos(max(-1.0, min(1.0, w)))
            s = math.sqrt(max(1e-12, 1.0 - w*w))
            ax, ay, az = x/s, y/s, z/s
            return [ax, ay, az, ang]

        def slerp(q0, q1, t):
            w0,x0,y0,z0 = q0; w1,x1,y1,z1 = q1
            dot = w0*w1 + x0*x1 + y0*y1 + z0*z1
            # assegura menor arco
            if dot < 0.0:
                w1, x1, y1, z1 = -w1, -x1, -y1, -z1
                dot = -dot
            if dot > 0.9995:  # quase linear
                w = w0 + t*(w1-w0); x = x0 + t*(x1-x0); y = y0 + t*(y1-y0); z = z0 + t*(z1-z0)
                n = math.sqrt(w*w + x*x + y*y + z*z) or 1.0
                return (w/n, x/n, y/n, z/n)
            th = math.acos(dot)
            s0 = math.sin((1.0-t)*th)/math.sin(th)
            s1 = math.sin(t*th)/math.sin(th)
            return (w0*s0 + w1*s1, x0*s0 + x1*s1, y0*s0 + y1*s1, z0*s0 + z1*s1)

        a0 = R[i]; a1 = R[i+1]
        q0 = axis_angle_to_q(*a0); q1 = axis_angle_to_q(*a1)
        q  = slerp(q0, q1, u)
        return q_to_axis_angle(q)

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
