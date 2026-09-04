# Método 0: Calibração Atual por AprilTags Pontuais (Baseline)

## Visão Geral
Este método representa a abordagem base utilizada até o momento no projeto. Ele baseia-se na detecção visual de marcadores AprilTag (família `tag36h11`) e na correspondência de raios visuais 3D com a odometria e nuvem de pontos do LiDAR Vanjee 722z.

## Componentes Arquiteturais:
1. **Lever Arm Físico Fixo**:
   - A câmera Insta360 X4 está montada a $18.5\text{ cm}$ acima da cabeça ótica do LiDAR ao longo do vetor vertical de gravidade do punho:
     $$\mathbf{c}_L = 0.185 \cdot \vec{u}_L = [0.0006,\ -0.0934,\ -0.1597]\text{ metros}$$
2. **Sincronização Temporal com Giroscópio (IMU INSV)**:
   - Sincronização em milissegundos via correlação cruzada do perfil de velocidade angular do giroscópio da Insta360 (1000 Hz) com o IMU do LiDAR (200 Hz):
     $$\Delta t = 5.7075\text{ s}$$ (coeficiente de Pearson $r = 0.9996$).
3. **Projeção da Câmera**:
   - Modelo simplificado puramente linear equidistante: $r = f \cdot \theta$, com $f = (W/2) / (\text{fov\_rad}/2)$ e $\text{fov} = 196.0^\circ$.
4. **Solução Numérica**:
   - Triangulação dos centros dos marcadores no espaço 3D global do SLAM.
   - Minimização do erro de distância raio-alvo (Levenberg-Marquardt) para estimar os ângulos de rotação [pitch, yaw, roll].

## Limitações Identificadas na Pesquisa:
- **Ausência de modelo polinomial de distorção de lente**: A lente da Insta360 X4 apresenta severa distorção radial periférica (não-linear), fazendo com que um erro subpíxel na imagem se expanda em um grande erro métrico tridimensional em distâncias > 3 metros.
- **Esparsidade do LiDAR 16 linhas**: Em 16 canais rotativos, a probabilidade de um feixe laser atingir o centro exato de um marcador AprilTag de 15 cm a distâncias médias é baixa, induzindo ruído estocástico na amostragem.
- **Diferença de paralaxe observada**: Molduras de quadros e portas na parede apresentavam pequenos deslocamentos visuais de textura ($1^\circ$ a $2^\circ$ de pitch/yaw), necessitando de ajustes manuais posteriores.
