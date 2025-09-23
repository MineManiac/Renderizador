#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time         # Para operações com tempo
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

        rgb_flat = GL._get_rgb_from_colors(colors)
        alpha = 1.0 - float(colors.get('transparency', 0.0)) if isinstance(colors, dict) else 1.0
        assert len(point) % 9 == 0, "TriangleSet espera múltiplos de 9 (x,y,z por vértice)"
        for i in range(0, len(point), 9):
            a = GL._project_point(point[i+0], point[i+1], point[i+2])
            b = GL._project_point(point[i+3], point[i+4], point[i+5])
            c = GL._project_point(point[i+6], point[i+7], point[i+8])
            if a and b and c:
                # sem cores por vértice aqui → usa flat em c0 e None nas outras
                GL._raster_triangle(a, b, c, rgb_flat, None, None, base_alpha=alpha)


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
    

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
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

        rgb_flat = GL._get_rgb_from_colors(colors or {})
        alpha = 1.0 - float((colors or {}).get('transparency', 0.0))

        verts = []
        if coord:
            verts = [(coord[i], coord[i+1], coord[i+2]) for i in range(0, len(coord), 3)]

        # paleta de cores por vértice (0..255)
        vcolors = None
        if color:
            vc = []
            for i in range(0, len(color), 3):
                r,g,b = color[i], color[i+1], color[i+2]
                if 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0:
                    vc.append([int(r*255), int(g*255), int(b*255)])
                else:
                    vc.append([int(r), int(g), int(b)])
            vcolors = vc

        use_ci = bool(colorPerVertex and vcolors and colorIndex)

        # texcoords
        tcoords = None
        if texCoord is not None:
            if isinstance(texCoord, (list, tuple)):
                arr = texCoord
            elif hasattr(texCoord, "point"):
                arr = texCoord.point
            else:
                arr = None
            if arr:
                tcoords = [(arr[i], arr[i+1]) for i in range(0, len(arr), 2)]
        use_ti = bool(tcoords is not None and texCoordIndex is not None)

        # handle de textura (pode ser int, lista, dict, etc.)
        tex_handle = GL._get_texture_handle(current_texture)

        face, fcols, fuvs = [], [], []
        ci_pos = 0
        ti_pos = 0

        def flush_face():
            if len(face) < 3:
                return
            for i in range(2, len(face)):
                i0, i1, i2 = face[0], face[i-1], face[i]
                a = GL._project_point(*verts[i0])
                b = GL._project_point(*verts[i1])
                c = GL._project_point(*verts[i2])
                if not (a and b and c): 
                    continue

                # cores nos vértices (opcional)
                if use_ci and fcols:
                    c0 = vcolors[fcols[0]]; c1 = vcolors[fcols[i-1]]; c2 = vcolors[fcols[i]]
                elif colorPerVertex and vcolors:
                    c0 = vcolors[i0]; c1 = vcolors[i1]; c2 = vcolors[i2]
                else:
                    c0 = rgb_flat; c1 = None; c2 = None

                # UVs por vértice (opcional)
                if tex_handle is not None and tcoords is not None:
                    if use_ti and fuvs:
                        uv0 = tcoords[fuvs[0]]; uv1 = tcoords[fuvs[i-1]]; uv2 = tcoords[fuvs[i]]
                    else:
                        # sem texCoordIndex → segue coordIndex
                        uv0 = tcoords[i0] if i0 < len(tcoords) else (0.0,0.0)
                        uv1 = tcoords[i1] if i1 < len(tcoords) else (0.0,0.0)
                        uv2 = tcoords[i2] if i2 < len(tcoords) else (0.0,0.0)
                    # modula pela cor de vértice se existir; senão passa None para usar só textura
                    mod_cols = None if (c1 is None or c2 is None) else [c0, c1, c2]
                    GL._raster_triangle_tex(a, b, c, uv0, uv1, uv2, tex_handle, mod_cols, base_alpha=alpha)
                else:
                    GL._raster_triangle(a, b, c, c0, c1, c2, base_alpha=alpha)

        for idx_i, idx in enumerate(coordIndex or []):
            # consumimos colorIndex/texCoordIndex em paralelo (incluindo -1)
            col_idx = None
            if use_ci and ci_pos < len(colorIndex):
                col_idx = colorIndex[ci_pos]; ci_pos += 1
            t_idx = None
            if use_ti and ti_pos < len(texCoordIndex):
                t_idx = texCoordIndex[ti_pos]; ti_pos += 1

            if idx == -1:
                flush_face()
                face, fcols, fuvs = [], [], []
                continue

            face.append(int(idx))
            if use_ci and col_idx is not None and col_idx != -1:
                fcols.append(int(col_idx))
            if use_ti and t_idx is not None and t_idx != -1:
                fuvs.append(int(t_idx))

        print("current_texture:", type(current_texture), current_texture, "handle:", tex_handle)
        # última face
        flush_face()




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

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

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

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

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

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

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

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
