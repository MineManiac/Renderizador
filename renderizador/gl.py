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
    # === Projeto 1.2 (helpers & state) ===
    # Estados do pipeline 3D
    _model_stack = [  # pilha de matrizes de modelo
        [1,0,0,0,
         0,1,0,0,
         0,0,1,0,
         0,0,0,1]
    ]
    _view = [1,0,0,0,
             0,1,0,0,
             0,0,1,0,
             0,0,0,1]
    _proj = [1,0,0,0,
             0,1,0,0,
             0,0,1,0,
             0,0,0,1]
    
    # --- Matrizes 4x4 em arranjo linha-continua (row-major) ---
    @staticmethod
    def _m4_mul(a, b):
        c = [0]*16
        for r in range(4):
            for cidx in range(4):
                c[4*r+cidx] = (a[4*r+0]*b[0+cidx] +
                               a[4*r+1]*b[4*1+cidx] +
                               a[4*r+2]*b[4*2+cidx] +
                               a[4*r+3]*b[4*3+cidx])
        return c
    
    @staticmethod
    def _m4_vec(a, v):
        # v: [x,y,z,w]
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
        # normaliza eixo
        import math
        l = math.sqrt(ax*ax + ay*ay + az*az) or 1.0
        x, y, z = ax/l, ay/l, az/l
        c = math.cos(t); s = math.sin(t); ic = 1.0 - c
        # matriz 3x3 embutida em 4x4
        return [x*x*ic + c,     x*y*ic - z*s, x*z*ic + y*s, 0,
                y*x*ic + z*s,   y*y*ic + c,   y*z*ic - x*s, 0,
                z*x*ic - y*s,   z*y*ic + x*s, z*z*ic + c,   0,
                0,              0,            0,            1]
    
    @staticmethod
    def _m4_inverse_rigid(mat):
        # inverte uma matriz rígida T*R (sem escala não-uniforme)
        # extrai R (3x3) e t (3x1)
        R = [mat[0],mat[1],mat[2],
             mat[4],mat[5],mat[6],
             mat[8],mat[9],mat[10]]
        t = [mat[3], mat[7], mat[11]]
        # R^-1 = R^T
        Rt = [R[0],R[3],R[6],
              R[1],R[4],R[7],
              R[2],R[5],R[8]]
        # -R^T t
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



    @staticmethod
    def triangleSet(point, colors):
        """Projeto 1.2 — TriangleSet 3D preenchido (constante, sem z-buffer)."""
        import math
        rgb = GL._get_rgb_from_colors(colors)

        def ndc_to_screen(x_ndc, y_ndc):
            # NDC [-1,1] -> pixel; y cresce para baixo
            sx = (x_ndc * 0.5 + 0.5) * (GL.width  - 1)
            sy = (-y_ndc * 0.5 + 0.5) * (GL.height - 1)
            return sx, sy  # float (vamos fazer floor/ceil na bounding box)

        def edge(ax, ay, bx, by, cx, cy):
            # função de aresta orientada
            return (cy - ay) * (bx - ax) - (cx - ax) * (by - ay)

        assert len(point) % 9 == 0, "TriangleSet: múltiplo de 9 (x,y,z por vértice)"
        M   = GL._model_stack[-1]
        VP  = GL._m4_mul(GL._proj, GL._view)
        MVP = GL._m4_mul(VP, M)

        for i in range(0, len(point), 9):
            v0 = [point[i+0], point[i+1], point[i+2], 1.0]
            v1 = [point[i+3], point[i+4], point[i+5], 1.0]
            v2 = [point[i+6], point[i+7], point[i+8], 1.0]

            # espaço de recorte (clip)
            c0 = GL._m4_vec(MVP, v0)
            c1 = GL._m4_vec(MVP, v1)
            c2 = GL._m4_vec(MVP, v2)

            # evita w=0 (sem clipping por enquanto)
            if c0[3] == 0 or c1[3] == 0 or c2[3] == 0:
                continue

            # NDC
            x0, y0 = c0[0]/c0[3], c0[1]/c0[3]
            x1, y1 = c1[0]/c1[3], c1[1]/c1[3]
            x2, y2 = c2[0]/c2[3], c2[1]/c2[3]

            # viewport
            X0, Y0 = ndc_to_screen(x0, y0)
            X1, Y1 = ndc_to_screen(x1, y1)
            X2, Y2 = ndc_to_screen(x2, y2)

            # área; pula degenerado
            A = edge(X0, Y0, X1, Y1, X2, Y2)
            if A == 0:
                continue

            # bounding box (clamp nos limites da tela)
            xmin = max(0,              int(math.floor(min(X0, X1, X2))))
            xmax = min(GL.width  - 1,  int(math.ceil (max(X0, X1, X2))))
            ymin = max(0,              int(math.floor(min(Y0, Y1, Y2))))
            ymax = min(GL.height - 1,  int(math.ceil (max(Y0, Y1, Y2))))

            # varredura por baricêntricas (edge function)
            for y in range(ymin, ymax+1):
                for x in range(xmin, xmax+1):
                    w0 = edge(X0, Y0, X1, Y1, x, y)
                    w1 = edge(X1, Y1, X2, Y2, x, y)
                    w2 = edge(X2, Y2, X0, Y0, x, y)
                    inside_pos = (w0 >= 0) and (w1 >= 0) and (w2 >= 0)
                    inside_neg = (w0 <= 0) and (w1 <= 0) and (w2 <= 0)
                    if inside_pos or inside_neg:
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, rgb)


    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Coleta Viewpoint e constrói view/projection (Projeto 1.2)."""
        # Camera transform C = T(eye) * R(axis,angle)
        eye = position or [0.0,0.0,0.0]
        ax, ay, az, t = orientation if orientation else [0.0,0.0,1.0,0.0]
        T = GL._m4_translation(eye[0], eye[1], eye[2])
        R = GL._m4_rotation_axis_angle(ax, ay, az, t)
        C = GL._m4_mul(T, R)
        # View is inverse of camera transform (rigid)
        GL._view = GL._m4_inverse_rigid(C)
        # Projection
        aspect = GL.width / GL.height if GL.height else 1.0
        GL._proj = GL._perspective(fieldOfView, aspect, GL.near, GL.far)


    @staticmethod
    def transform_in(translation, scale, rotation):
        """Empilha Model = Parent * (T * R * S)."""
        parent = GL._model_stack[-1]
        # constrói T, R, S
        tx,ty,tz = (translation if translation else [0.0,0.0,0.0])
        sx,sy,sz = (scale if scale else [1.0,1.0,1.0])
        if rotation:
            rx,ry,rz,ang = rotation
            R = GL._m4_rotation_axis_angle(rx,ry,rz,ang)
        else:
            R = GL._m4_identity()
        T = GL._m4_translation(tx,ty,tz)
        S = GL._m4_scale(sx,sy,sz)
        local = GL._m4_mul(GL._m4_mul(T, R), S)  # T * R * S
        GL._model_stack.append(GL._m4_mul(parent, local))


    @staticmethod
    def transform_out():
        """Desempilha a matriz de modelo atual."""
        if len(GL._model_stack) > 1:
            GL._model_stack.pop()


    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TriangleStripSet : pontos = {0} ".format(point), end='')
        for i, strip in enumerate(stripCount):
            print("strip[{0}] = {1} ".format(i, strip), end='')
        print("")
        print("TriangleStripSet : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("IndexedTriangleStripSet : pontos = {0}, index = {1}".format(point, index))
        print("IndexedTriangleStripSet : colors = {0}".format(colors)) # imprime as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão não possui uma ordem oficial, mas em geral se o primeiro ponto com os dois
        # seguintes e depois este mesmo primeiro ponto com o terçeiro e quarto ponto. Por exemplo: numa
        # sequencia 0, 1, 2, 3, 4, -1 o primeiro triângulo será com os vértices 0, 1 e 2, depois serão
        # os vértices 0, 2 e 3, e depois 0, 3 e 4, e assim por diante, até chegar no final da lista.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.

        # Os prints abaixo são só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("IndexedFaceSet : ")
        if coord:
            print("\tpontos(x, y, z) = {0}, coordIndex = {1}".format(coord, coordIndex))
        print("colorPerVertex = {0}".format(colorPerVertex))
        if colorPerVertex and color and colorIndex:
            print("\tcores(r, g, b) = {0}, colorIndex = {1}".format(color, colorIndex))
        if texCoord and texCoordIndex:
            print("\tpontos(u, v) = {0}, texCoordIndex = {1}".format(texCoord, texCoordIndex))
        if current_texture:
            image = gpu.GPU.load_texture(current_texture[0])
            print("\t Matriz com image = {0}".format(image))
            print("\t Dimensões da image = {0}".format(image.shape))
        print("IndexedFaceSet : colors = {0}".format(colors))  # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

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
