# Método 0 Aprimorado: Modelo Polinomial Fisheye (Kannala-Brandt) + Sincronização Giroscópica

## 1. O Que a Pesquisa Trouxe para o Método Atual
O relatório científico [LiDAR Fisheye Calibration Research.md](file:///c:/Users/User/Documents/APLICATIVOS/Lidar-Camera-calibrator/docs/LiDAR%20Fisheye%20Calibration%20Research.md) aponta na Seção 3.1:
> *"O modelo perspetivo clássico e a aproximação linear pura equidistante falham em ângulos marginais próximos de 180°+ devido ao encurvamento extremo radial (barrel distortion). Um mero erro subpíxel na imagem fisheye expande-se por projeção inversa num erro métrico grave à medida que a distância do robô ao alvo aumenta."*

No método original, utilizava-se a fórmula de livro-texto $r = f \cdot \theta$. Implementamos o modelo polinomial de Kannala-Brandt de 4ª ordem ($k_1 = -0.04268, k_2 = 0.00242$):
$$\theta_d = \theta (1 + k_1 \theta^2 + k_2 \theta^4), \quad r = f \cdot \theta_d$$

---

## 2. Comparativo Numérico no Dataset Dinâmico (`DinamicAprilTagCalib`)
Avaliamos 218 observações multi-visão de AprilTags e a reprojeção métrica da moldura do quadro físico da parede a $6.55\text{ m}$ de distância:

| Métrica | Modelo Linear (Baseline Original) | Modelo Polinomial (Aprimorado) | Impacto |
| :--- | :--- | :--- | :--- |
| **Sincronização Temporal ($\Delta t$)** | $5.7075\text{ s}$ (Giroscópio IMU) | $5.7075\text{ s}$ (Giroscópio IMU) | Idêntico |
| **Lever Arm Físico ($\mathbf{c}_L$)** | $[0.0006, -0.0934, -0.1597]\text{ m}$ | $[0.0006, -0.0934, -0.1597]\text{ m}$ | 18.5 cm vertical mantido |
| **Pitch Trim Estimado** | $-0.4430^\circ$ | **$-1.3704^\circ$** | **Aproxima-se de Cand 07 ($-1.24^\circ$)** |
| **Yaw Trim Estimado** | $+0.6075^\circ$ | **$+0.2965^\circ$** | Convergência suave |
| **Desvio Horizontal na Moldura ($\Delta U$)** | $+12.9\text{ px}$ | **$+6.7\text{ px}$** | **Redução de 48% no erro horizontal** |
| **Desvio Vertical na Moldura ($\Delta V$)** | $-147.0\text{ px}$ | **$-128.9\text{ px}$** | **18.2 pixels corrigidos pela ótica** |

---

## 3. Conclusão Técnica
A incorporação do polinômio ótico explicou grande parte da divergência de inclinação observada nos testes de nuvem de pontos: o modelo linear subestimava a compressão radial da lente nas margens. Otimizar a rotação com a física da lente reduz os desvios periféricos sem necessidade de forçar trims manuais ad-hoc.
