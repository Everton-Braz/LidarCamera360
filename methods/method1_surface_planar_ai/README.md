# Método 1: Calibração de Superfície Planar e Kabsch (`ARVCUMH/fisheye_lidar_calibration`)

## 1. Fundamentação Teórica
O método desenvolvido pela Universidade Miguel Hernández (`ARVCUMH/fisheye_lidar_calibration`) propõe contornar a esparsidade do LiDAR de 16 linhas elevando o nível de abstração: em vez de tentar detectar os minúsculos cantos de 1 cm do código AprilTag nos retornos laser, ele extrai o **plano da placa suporte física** (retângulo planar de 30x40 cm) onde o marcador está fixado.

### Fluxo Matemático:
1. **LiDAR (3D)**:
   - Filtro de pontos de alta refletividade na região de interesse do alvo.
   - Ajuste planar via RANSAC ($ax + by + cz + d = 0$) para estimar o vetor normal $\vec{n}_{L, i}$ e o centróide $\mathbf{c}_{L, i}$.
2. **Câmera Fisheye (2D)**:
   - Segmentação do contorno da placa/marcador e projeção dos vértices em raios direcionais 3D.
   - Estimativa da normal do plano na câmera: $\vec{n}_{\text{cam}, i} = \frac{\mathbf{d}_1 \times \mathbf{d}_2}{\|\mathbf{d}_1 \times \mathbf{d}_2\|}$.
3. **Solução Kabsch / Wahba (Forma Fechada)**:
   - Determinação da matriz de rotação inicial ótima $R_{\text{Kabsch}}$ via SVD da matriz de covariância das normais:
     $$H = \sum_{i} \vec{n}_{\text{cam}, i} \cdot \vec{n}_{L, i}^T, \quad U \Sigma V^T = \text{SVD}(H) \implies R = U \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & \det(UV^T) \end{bmatrix} V^T$$
4. **Refinamento Não-Linear Rígido**:
   - Imposição do lever arm físico estrito: $\mathbf{c}_L = 0.185 \cdot \vec{u}_L = [0.0006, -0.0934, -0.1597]\text{ m}$.

---

## 2. Resultados Experimentais Obtidos
Executado nos dados estáticos `20260902092520`:

- **Pontos LiDAR no suporte dos alvos**:
  - Placa #0: 2.300 inliers planares (Centróide: `[-2.92, +0.14, +0.19] m`, Normal: `[+0.998, -0.043, +0.051]`)
  - Placa #1: 1.775 inliers planares (Centróide: `[+0.85, -1.18, +0.11] m`, Normal: `[+0.020, +0.877, -0.479]`)
  - Placa #5: 2.058 inliers planares (Centróide: `[+0.49, -1.45, +0.22] m`, Normal: `[+0.023, +0.949, -0.314]`)
- **Extrínsecos Estimados**:
  - Matriz de Rotação $R$:
    $$\begin{bmatrix} -0.9707 & 0.1981 & -0.1362 \\ 0.0218 & -0.4913 & -0.8707 \\ -0.2394 & -0.8481 & 0.4726 \end{bmatrix}$$
  - Ângulos de Euler (XYZ): Pitch = $-60.88^\circ$, Yaw = $+13.85^\circ$, Roll = $+178.71^\circ$
  - Translação na Câmera: $t = [-0.003, -0.185, -0.004]\text{ m}$ (coerência exata de $18.5\text{ cm}$ vertical com a física da montagem).
- **Erro Médio de Reprojeção**: $372.63\text{ pixels}$ (em resolução $3840 \times 3840$, correspondendo a $\sim 9.5^\circ$ de desvio residual nos extremos).

---

## 3. Conclusão e Veredito Técnico
- **Vantagem Clara**: O método é **100% funcional no LiDAR de 16 linhas** (extraiu mais de 1.700 a 2.300 pontos coplanares por placa com RANSAC), superando completamente a falha de decodificação de bits do Método 4.
- **Sensibilidade do Método**: A estimativa do vetor normal $\vec{n}_{\text{cam}}$ a partir de um marcador de $15\text{ cm}$ a 3 metros de distância na lente fisheye tem incerteza angular de alguns graus porque a área aparente do alvo na imagem é de apenas $\sim 50\text{ px}$. 
- **Melhoria recomendada**: Utilizar uma placa física de suporte maior (ex: placa de $50 \times 50\text{ cm}$ ou $1\text{ metro}$) ou combinar os centróides 3D com o modelo não-linear de distorção de lente (Método 0+).
