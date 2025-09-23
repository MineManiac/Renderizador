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
Para rodar alguns dos exemplos de validação:

```bash
# pontos
python3 renderizador/renderizador.py -i docs/exemplos/2D/pontos/aleatorios/aleatorios.x3d -w 600 -h 400 -p

# linhas
python3 renderizador/renderizador.py -i docs/exemplos/2D/linhas/linhas_cores/linhas_cores.x3d -w 600 -h 400 -p

# triângulos
python3 renderizador/renderizador.py -i docs/exemplos/2D/triangulos/triangulos/triangulos.x3d -w 600 -h 400 -p
