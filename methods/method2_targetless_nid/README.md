# Método 2: Calibração Target-less por Distância de Informação Normalizada (NID)

## 1. Fundamentação Teórica
Proposto por Kenji Koide et al. (`koide3/direct_visual_lidar_calibration`), este método elimina a necessidade de qualquer alvo fiduciário (AprilTags ou tabuleiros de xadrez). Ele alinha a geometria tridimensional do ambiente capturada pelo LiDAR (quebras de normais de superfície e contrastes de refletividade) diretamente com os gradientes fotométricos da imagem da câmera, maximizando a Informação Mútua ($I(X; Y)$) e minimizando a Distância de Informação Normalizada (NID):

$$\text{NID}(X, Y) = 1 - \frac{I(X; Y)}{H(X, Y)} = \frac{2 H(X, Y) - H(X) - H(Y)}{H(X, Y)}$$

Onde $X$ representa o ângulo de incidência/refletividade do ponto laser e $Y$ representa a intensidade e gradiente de borda na imagem 2D.

---

## 2. Resultados Experimentais Obtidos
Executado sobre o ambiente estático `20260902092520` (sala completa com paredes, portas, quadros e teto):

- **Pontos LiDAR Amostrados**: 20.000 pontos com cálculo de normais e vetores de visão.
- **Câmera**: Lente frontal $3840 \times 3840$ com modelo de lente não-linear Kannala-Brandt.
- **Otimizador**: Powell Multivariável sobre $[p, y, r]$ mantendo o lever arm físico rígido ($\mathbf{c}_L = 0.185 \cdot \vec{u}_L$).

### Métricas de Convergência:
| Métrica | Valor Inicial | Valor Otimizado | Variação |
| :--- | :--- | :--- | :--- |
| **Custo NID** | $0.95671$ | **$0.95287$** | Redução de entropia cruzada |
| **Pitch Trim** | $0.0000^\circ$ | **$+0.1294^\circ$** | Mínima variação vertical |
| **Yaw Trim** | $0.0000^\circ$ | **$-1.5962^\circ$** | **Coincide com medição física do usuário ($-1.5^\circ$ a $-2.0^\circ$)!** |
| **Roll Trim** | $0.0000^\circ$ | **$-0.0675^\circ$** | Estabilidade em torno do eixo óptico |
| **Translação na Câmera** | $[0, -0.185, 0]\text{ m}$ | $[-0.0002, -0.1850, -0.0004]\text{ m}$ | Perfeita preservação métrica |

---

## 3. Conclusão e Veredito Técnico
- **Autonomia Total**: Não depende de alvos impressos, iluminação infravermelha em marcadores pequenos ou calibração de cantos.
- **Validação Cruzada Notável**: O otimizador NID convergiu de forma puramente estatística para um **Yaw Trim de $-1.60^\circ$**, reproduzindo com precisão a medição de desvio horizontal efetuada manualmente pelo usuário no CloudCompare no frame da parede!
- **Recomendação de Uso**: Excelente como etapa final de refinamento em qualquer ambiente com texturas e arestas arquitetônicas (quinas de paredes, batentes de portas, móveis).
