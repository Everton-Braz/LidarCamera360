# Método 4: Deteção Cruzada de AprilTag em Imagem de Intensidade LiDAR (`srrg2_apriltag_calibration`)

## 1. Fundamentação Teórica
O método clássico proposto pela Universidade de Roma (`rvp-group/srrg2_apriltag_calibration`) projeta a nuvem de pontos LiDAR em uma "imagem sintética de intensidade 2D" (raster cilíndrico/esférico baseado em azimute e elevação). Em seguida, executa a biblioteca AprilTag3 simultaneamente na imagem da câmera e na imagem de intensidade do LiDAR, estabelecendo correspondências 2D-2D para estimar a pose extrínseca via erro de reprojeção.

## 2. Teste Empírico no Vanjee 722z (16 Linhas Rotativas)
Executamos a projeção panorâmica cilíndrica de alta resolução ($4096 \times 1024$) nos dados do sensor estático `20260902092520`:

### Resultados Obtidos:
| Sensor / Modalidade | Resolução Efetiva | Tags Detectados | Observações |
| :--- | :--- | :--- | :--- |
| **Câmera Insta360 X4 (Frontal)** | $3840 \times 3840$ | **3 Tags** [0, 1, 5] | Detecção subpíxel imediata |
| **Câmera Insta360 X4 (Traseira)** | $3840 \times 3840$ | **1 Tag** [0] | Detecção nítida na borda periférica |
| **LiDAR Vanjee 722z (Raster Esparso)** | $4096 \times 1024$ | **0 Tags** | Bits binários não reconstituídos |
| **LiDAR Vanjee 722z (Raster Interpolado)** | $4096 \times 1024$ | **0 Tags** (42 rejeitados) | Esparsidade angular crítica |

### Análise Numérica de Feixes vs. Dimensões do Marcador (15x15 cm):
- **Divergência angular vertical ($\Delta \theta_v$)**: $\sim 2.0^\circ$ entre anéis consecutivos.
- **Tag 0 (Distância 1.85 m)**: Espaçamento entre feixes de $6.5\text{ cm} \rightarrow$ **Apenas 2.3 linhas laser** interceptam o alvo.
- **Tag 1 (Distância 2.40 m)**: Espaçamento entre feixes de $8.4\text{ cm} \rightarrow$ **Apenas 1.8 linhas laser** interceptam o alvo.
- **Tag 5 (Distância 3.10 m)**: Espaçamento entre feixes de $10.8\text{ cm} \rightarrow$ **Apenas 1.4 linhas laser** interceptam o alvo.

## 3. Conclusão Científica
A decodificação de um AprilTag da família `tag36h11` exige a amostragem intacta de pelo menos 10 células de largura (borda preta de 1 bit + borda branca de 1 bit + payload 6x6 bits + margem). Com apenas 1 a 2 feixes interceptando a área do marcador, a recuperação dos bits binários no LiDAR é matematicamente impossível. 

**Veredito do Método 4**: **Inviável para LiDARs de 16 feixes** com alvos de tamanho padrão ($15\text{ cm}$), confirmando com 100% de exatidão a previsão do relatório científico. Para funcionar neste método, seria obrigatório imprimir alvos com mais de 1.5 metro de largura.
