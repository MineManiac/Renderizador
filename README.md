# Renderizador

# Projeto 1.1 — Rasterização 2D

## Implementações
Foram implementadas as seguintes funções em `gl.py`:
- **polypoint2D**: renderiza pontos recebendo pares (x, y).
- **polyline2D**: renderiza segmentos de reta conectando pontos consecutivos, usando o algoritmo de Bresenham.
- **triangleSet2D**: renderiza triângulos 2D, desenhando suas arestas com Bresenham.

## Decisões de Implementação
- Adicionei checagem de limites antes de desenhar (`if 0 <= x < width and 0 <= y < height`) para evitar acessos inválidos ao framebuffer.
- As cores são normalizadas de acordo com o campo `emissiveColor` (valores 0..1 convertidos para 0..255).
- Triângulos por enquanto são desenhados apenas pelas **arestas** (preenchimento será feito em projetos posteriores).

## Como Executar
Para rodar alguns dos exemplos de validação do 1.1:

```bash
# pontos
python3 renderizador/renderizador.py -i docs/exemplos/2D/pontos/aleatorios/aleatorios.x3d -w 600 -h 400 -p

# linhas
python3 renderizador/renderizador.py -i docs/exemplos/2D/linhas/linhas_cores/linhas_cores.x3d -w 600 -h 400 -p

# triângulos
python3 renderizador/renderizador.py -i docs/exemplos/2D/triangulos/triangulos/triangulos.x3d -w 600 -h 400 -p

```


# Projeto 1.2 — Pipeline 3D básico

## Implementações
Foram implementadas as seguintes funções em `gl.py`:
- **triangleSet**: rasteriza triângulos **3D preenchidos** após transformar para tela (MVP → NDC → viewport).
- **viewpoint**: constrói a matriz de **View** como a inversa rígida de `C = T(eye) · R(axis,angle)` e define a **Projection** perspectiva (`fieldOfView`, `aspect`, `near`, `far`).
- **transform_in**: compõe e **empilha** a matriz de **Model** com `T · R · S` (ordem do nó), respeitando a hierarquia do grafo de cena.

## Decisões de Implementação
- **Pilha de matrizes** para o Model; ordem de composição: `Model = Parent · (T · R · S)`.
- **Viewport mapping**:  
  `x_screen = (x_ndc*0.5 + 0.5) * (width - 1)`  
  `y_screen = (-y_ndc*0.5 + 0.5) * (height - 1)` (inversão de Y para a tela).
- **Rasterização** por **função de aresta** (teste baricêntrico por sinal) em coordenadas de tela.
- **Limitações nesta etapa**: sem z-buffer, sem texturas/transparência e sem anti-aliasing.

## Como Executar
Para rodar os exemplos de validação do 1.2:

```bash
# um triângulo 3D (sólido)
python3 renderizador/renderizador.py -i docs/exemplos/3D/malhas/um_triangulo/um_triangulo.x3d -w 800 -h 600 -p

# vários triângulos (testa T·R·S e hierarquia)
python3 renderizador/renderizador.py -i docs/exemplos/3D/malhas/varios_triangs/varios_triangs.x3d -w 800 -h 600 -p

# zoom (testa fieldOfView do Viewpoint)
python3 renderizador/renderizador.py -i docs/exemplos/3D/malhas/zoom/zoom.x3d -w 800 -h 600 -p
```
