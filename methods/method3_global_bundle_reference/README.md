# Método 3: Ajuste de Blocos com Referencial Global Estático (`PRBonn/ipb_calibration`)

## 1. Fundamentação Teórica
O método de Ajuste Conjunto de Blocos desenvolvido pela Universidade de Bonn (`PRBonn/ipb_calibration`) substitui a dependência de um único alvo ou medição pontual estática por uma **rede de restrições globais**. 

O sistema utiliza a trajetória tridimensional completa de odometria/SLAM (gerada pelo FAST-LIVO2) como um sistema métrico contínuo. Durante o percurso, múltiplos alvos (Tags #0 a #5) espalhados pela sala são observados em dezenas de ângulos, distâncias e iluminações diferentes. A otimização conjunta estima simultaneamente:
1. Os 6 graus de liberdade (DoF) extrínsecos entre LiDAR e Câmera.
2. A sincronização temporal $\Delta t$.
3. As posições 3D cartesianas $\mathbf{X}_w = (X, Y, Z)$ de cada marcador no referencial global do mundo.

Ao integrar 235 observações ao longo de 90 segundos de movimento contínuo, a esparsidade momentânea do LiDAR de 16 feixes é completamente compensada pelo acúmulo espaço-temporal da nuvem.

---

## 2. Resultados Experimentais Obtidos
Executado sobre o dataset dinâmico `DinamicAprilTagCalib`:

- **Observações Processadas**: 235 detecções multi-visão dos Tags 0, 1, 2, 3, 4 e 5.
- **Odometria Métrica**: 300 poses SLAM com quaternions e posições $t \in [0, 89.7\text{ s}]$.
- **Modelo de Câmera**: Polinomial Kannala-Brandt ($k_1 = -0.0427, k_2 = 0.0024$).
- **Lever Arm Físico**: $\mathbf{c}_L = 0.185 \cdot \vec{u}_L = [0.0006, -0.0934, -0.1597]\text{ m}$.

### Métricas Otimizadas:
| Parâmetro | Valor Otimizado | Impacto / Significado |
| :--- | :--- | :--- |
| **Pitch Trim** | **$-2.9723^\circ$** | Compensa a inclinação física da haste no suporte |
| **Yaw Trim** | **$+0.4254^\circ$** | Alinhamento azimutal fino |
| **Roll Trim** | **$+0.3596^\circ$** | Ajuste fino em torno do eixo óptico |
| **Sincronização Temporal ($\Delta t$)** | **$5.7075\text{ s}$** | Confirmação exata com o IMU Gyro da Insta360 |
| **Erro RMS 3D Raio-Alvo** | **$176.5\text{ mm}$** | Resíduo em alvos a até 6 metros |
| **Desvio Vertical na Moldura ($\Delta V$)** | **$-97.5\text{ px}$** | **Redução de 50 pixels em relação ao baseline ($-147\text{ px}$)** |
| **Desvio Horizontal na Moldura ($\Delta U$)** | **$+8.9\text{ px}$** | Menos de 9 pixels em $3840 \times 3840$ (< 0.23% da imagem) |

### Reconstrução 3D dos Alvos no Mundo:
- **Tag #0**: `[-0.063, -1.612, +0.124] m` (53 observações)
- **Tag #1**: `[+1.305, +0.810, -0.290] m` (29 observações)
- **Tag #2**: `[+0.374, +1.771, -0.086] m` (50 observações)
- **Tag #3**: `[-1.876, +0.003, -0.393] m` (46 observações)
- **Tag #4**: `[+5.465, -0.055, -0.049] m` (19 observações - corredor ao fundo)
- **Tag #5**: `[+1.686, +0.792, -0.146] m` (38 observações)

---

## 3. Conclusão Técnica
O Método 3 forneceu a convergência mais robusta e geometricamente estável de todos os métodos baseados em alvos fiduciários, pois a redundância de centenas de raios de posições diferentes impede que o otimizador caia em mínimos locais. A redução de 50 pixels no erro de paralaxe vertical da moldura atesta o poder da formulação global.
