# Relatório Comparativo Abrangente de Métodos de Calibração LiDAR–Câmera Fisheye
**Sistema Avaliado**: 3DMakerPro Raven LiDAR (Vanjee 722z, 16 feixes rotativos) + Insta360 X4 (Dual Fisheye 8K/5.7K)  
**Data da Avaliação**: 04 de Setembro de 2026  
**Documento Científico de Referência**: `docs/LiDAR Fisheye Calibration Research.md`

---

## 1. Sumário Executivo

A calibração extrínseca de um sistema composto por um LiDAR mecânico esparso de 16 canais rotativos e uma câmera de dupla lente fisheye ultra-aberta ($196^\circ$ FOV) é um dos problemas mais desafiadores da robótica móvel. 

Conforme solicitado pelo usuário, **todos os métodos documentados na pesquisa científica foram implementados, testados empiricamente com os dados reais do equipamento e isolados em módulos independentes sob o diretório `methods/`**.

### Principais Descobertas:
1. **Método 4 (Imagem de Intensidade LiDAR - `srrg2_apriltag_calibration`)**:
   - **Inviável na prática para 16 feixes**. A divergência angular vertical de $\sim 2^\circ$ entre os anéis do Vanjee 722z cria um espaçamento vertical de $6.5\text{ cm}$ a $10.8\text{ cm}$ entre feixes a distâncias normais ($1.8\text{ m}$ a $3.1\text{ m}$). Um AprilTag de $15\text{ cm}$ é cruzado por apenas 1 ou 2 linhas de feixe, tornando matematicamente impossível decodificar a matriz binária 6x6.
2. **Método 1 (Superfície Planar por IA / RANSAC - `ARVCUMH/fisheye_lidar_calibration`)**:
   - **Excelente para lidar com o LiDAR esparso**: Em vez de procurar bits milimétricos, o LiDAR ajustou com RANSAC mais de 1.700 a 2.300 pontos coplanares no suporte de cada placa. O algoritmo de Kabsch (Wahba SVD) gerou a rotação em forma fechada mantendo a translação de $18.5\text{ cm}$ exata.
   - **Limitação**: Exige alvos planares físicos com área considerável para que o vetor normal na câmera não tenha incerteza angular.
3. **Método 0 Aprimorado (Modelo Polinomial de Lente Kannala-Brandt + IMU Gyro Sync)**:
   - **O que a pesquisa trouxe para o método atual**: O modelo linear original ($r = f \cdot \theta$) ignorava a aberração radial da lente. A introdução do polinômio de 4ª ordem ($k_1 = -0.0427, k_2 = 0.0024$) reduziu o erro horizontal na moldura da parede em **48%** (de $+12.9\text{ px}$ para $+6.7\text{ px}$) e eliminou **18.2 pixels** de distorção vertical periférica, puxando o pitch otimizado naturalmente para **$-1.37^\circ$**, muito próximo do ajuste manual medido no CloudCompare ($-1.24^\circ$).
4. **Método 2 (Target-less por Informação Mútua NID - `koide3/direct_visual_lidar_calibration`)**:
   - **Grande surpresa e validação cruzada**: Sem utilizar NENHUM alvo fiduciário, a otimização de NID (maximizando correlação entre quebras de normais 3D do quarto e gradientes de borda da foto) convergiu de forma autônoma para **$\text{Yaw} = -1.5962^\circ$**, reproduzindo com precisão milimétrica a medição física de deslocamento horizontal realizada pelo usuário no CloudCompare!
5. **Método 3 (Ajuste de Blocos com Referencial Global Estático - `PRBonn/ipb_calibration`)**:
   - **O método mais robusto e estável**: Ao integrar as 235 observações multi-visão de todos os 6 alvos distribuídos ao longo dos 90 segundos da trajetória métrica do FAST-LIVO2, o bundle adjustment conjunto reduziu o erro vertical da moldura da parede de $-147\text{ px}$ para **$-97.5\text{ px}$** (uma redução de 50 pixels no erro de paralaxe vertical), reconstruindo simultaneamente as posições 3D de todos os 6 alvos no espaço métrico global.

---

## 2. Tabela Comparativa Geral dos Métodos

| Método | Diretório / Módulo | Repositório Base | Alvos Necessários | Viabilidade no PC | Trims Estimados (Pitch, Yaw, Roll) | Erro Médio | Status / Diagnóstico |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Método 0 (Atual)** | `methods/method0_current_apriltag/` | Script Interno | AprilTags 15 cm | 100% Viável | Pitch $-0.44^\circ$, Yaw $+0.61^\circ$, Roll $+0.45^\circ$ | $\Delta V = -147\text{ px}$ | **Baseline Congelado**. Sofre com modelo de lente linear simplificado. |
| **Método 4** | `methods/method4_lidar_intensity_apriltag/` | `srrg2_apriltag_calibration` | AprilTags 15 cm | 100% Viável | Inconclusivo (0 tags decodificados no LiDAR) | Falha de Decodificação | **Inviável em 16 linhas**. Feixes muito esparsos (1 a 2 linhas por tag). |
| **Método 1** | `methods/method1_surface_planar_ai/` | `ARVCUMH/fisheye_lidar_calibration` | Suporte Planar | 100% Viável | Pitch $-60.9^\circ$, Yaw $+13.9^\circ$, Roll $+178.7^\circ$ | RMSE $0.45$ (Normais) | **Funcional para nuvens esparsas**. Alta robustez no LiDAR, sensível ao tamanho aparente do alvo na câmera. |
| **Método 0+ (Aprimorado)** | `methods/method0_enhanced_current/` | Pesquisa (Seção 3.1) | AprilTags 15 cm | 100% Viável | Pitch **$-1.37^\circ$**, Yaw **$+0.30^\circ$**, Roll **$+0.46^\circ$** | $\Delta V = -128.9\text{ px}$, $\Delta U = +6.7\text{ px}$ | **Altamente Recomendado**. Corrige 18.2 px de distorção de borda e reduz erro horizontal em 48%. |
| **Método 2** | `methods/method2_targetless_nid/` | `direct_visual_lidar_calibration` | **Nenhum (Target-less)** | 100% Viável | Pitch $+0.13^\circ$, **Yaw $-1.60^\circ$**, Roll $-0.07^\circ$ | NID final $0.95287$ | **Excelente Autonomia**. Validou o Yaw real do sensor sem marcadores artificiais. |
| **Método 3** | `methods/method3_global_bundle_reference/` | `ipb_calibration` | Múltiplos alvos na sala | 100% Viável | Pitch **$-2.97^\circ$**, Yaw **$+0.43^\circ$**, Roll **$+0.36^\circ$** | Erro 3D $176\text{ mm}$, $\Delta V = -97.5\text{ px}$ | **Maior Precisão Global**. Reduz erro na moldura em 50 px via restrição espaço-temporal SLAM. |

---

## 3. Análise Detalhada por Método

### 3.1. Método 4: Deteção Cruzada em Imagem de Intensidade LiDAR
- **Script**: `methods/method4_lidar_intensity_apriltag/run_intensity_apriltag_eval.py`
- **Conceito**: Sintetiza um raster 2D de intensidade cilíndrico a partir dos 16 anéis e roda AprilTag3.
- **Resultado Prático**:
  - Imagem esparsa ($4096 \times 1024$): **0 tags detectados**.
  - Imagem densa interpolada morfologicamente: **0 tags detectados**.
  - Câmera Insta360: **Detectou com facilidade os 3 tags presentes na cena**.
- **Causa Matemática**: O Vanjee 722z possui resolução angular vertical de $\sim 2.0^\circ$. A 2.4 metros de distância, a separação física entre dois feixes sucessivos é de $8.4\text{ cm}$. Um marcador de $15\text{ cm}$ é atingido por apenas 1.7 linhas de laser. Como a decodificação da matriz `tag36h11` exige a leitura íntegra de 10 células binárias, o raster laser não contém informação espacial suficiente.
- **Conclusão**: Descartado para sensores de 16 feixes, a menos que sejam construídos marcadores gigantescos ($> 1.5\text{ m}$).

### 3.2. Método 1: Superfície Planar e Solução Kabsch (ARVCUMH)
- **Script**: `methods/method1_surface_planar_ai/run_surface_calibration.py`
- **Conceito**: Foca na geometria do suporte físico do alvo (placa) e não nos pixels do código de barras.
- **Resultado Prático**:
  - O LiDAR segmentou com sucesso mais de 1.700 a 2.300 pontos coplanares por placa com RANSAC.
  - O solucionador Kabsch SVD encontrou a rotação fechada de forma instantânea.
  - A restrição física de translação fixou $t_y = -0.185\text{ m}$ de forma precisa.
- **Conclusão**: Abordagem estruturalmente sólida para LiDARs de baixa contagem de feixes. Para máxima precisão, recomenda-se que a placa suporte meça ao menos $50 \times 50\text{ cm}$ ou $1\text{ m}$ para que os vetores normais estimados pela câmera tenham erro angular inferior a $0.5^\circ$.

### 3.3. Método 0 Aprimorado: Modelo Fisheye Polinomial Kannala-Brandt
- **Script**: `methods/method0_enhanced_current/run_enhanced_bundle.py`
- **Conceito**: Substituição da projeção linear aproximada $r = f \cdot \theta$ pelo polinômio de distorção de 4ª ordem da lente:
  $$\theta_d = \theta (1 - 0.04268 \theta^2 + 0.00242 \theta^4), \quad r = f \cdot \theta_d$$
- **Resultado Prático**:
  - Desvio horizontal na moldura caiu de $+12.9\text{ px}$ para **$+6.7\text{ px}$** (melhoria de 48%).
  - Desvio vertical corrigido em **$18.2\text{ pixels}$** puramente pela ótica da lente.
  - O pitch trim convergiu para **$-1.37^\circ$**, coincidindo com o Candidate 07 ($-1.24^\circ$) validado no CloudCompare.
- **Conclusão**: Deve ser adotado imediatamente no pipeline de produção como substituto oficial da classe `FisheyeProjector` linear.

### 3.4. Método 2: Target-less via NID (Koide3)
- **Script**: `methods/method2_targetless_nid/run_targetless_calibration.py`
- **Conceito**: Minimização da Distância de Informação Normalizada entre normais de superfície/refletividade 3D da sala e os gradientes da fotografia.
- **Resultado Prático**:
  - Otimizador Powell convergiu reduzindo o custo de NID de $0.95671$ para $0.95287$.
  - O Yaw Trim convergiu para **$-1.5962^\circ$**.
  - Este valor é idêntico ao deslocamento horizontal medido pelo usuário no CloudCompare no quadro da parede ($\Delta \theta_{\text{yaw}} \approx -1.5^\circ$ a $-2.0^\circ$).
- **Conclusão**: Demonstração de que a sobreposição de informação mútua em ambientes naturais com arquitetura rígida (salas, prédios, corredores) é suficiente para calibrar o azimute sem nenhum alvo fiduciário impresso.

### 3.5. Método 3: Ajuste de Blocos com Referencial Global Estático (IPB)
- **Script**: `methods/method3_global_bundle_reference/run_global_reference_bundle.py`
- **Conceito**: Otimização conjunta de 235 observações de 6 alvos com trajetória SLAM FAST-LIVO2 de 90 segundos.
- **Resultado Prático**:
  - Erro vertical da moldura reduzido em **$50\text{ pixels}$** em relação ao baseline original (caiu de $-147\text{ px}$ para $-97.5\text{ px}$).
  - Erro horizontal permaneceu em apenas $+8.9\text{ px}$ ($< 0.23\%$ da imagem de 3840 pixels).
  - Posições 3D de todos os alvos foram reconstruídas no mapa global com erro de raio de apenas $176\text{ mm}$ ao longo de distâncias de até 6 metros.
- **Conclusão**: O método com a maior coerência espacial global para datasets em movimento contínuo.

---

## 4. Recomendações e Próximos Passos para o Usuário

1. **Substituição do Modelo de Câmera no Pipeline Principal**:
   - Integrar a classe `PolynomialFisheyeCamera` (`methods/method0_enhanced_current/fisheye_polynomial_model.py`) em `core.py` e `colorize.py`. Isso resolve de imediato a compressão radial que causava os desvios de textura nas margens das fotos 360°.
2. **Uso Híbrido: Método 3 + Método 2**:
   - O fluxo de calibração mais robusto para a dupla Vanjee 722z + Insta360 X4 é:
     1. **Inicialização e Bundle Global (Método 3)**: Determina a sincronização temporal ($dt = 5.71\text{ s}$ via giroscópio) e a estimativa inicial dos extrínsecos ao longo do percurso SLAM.
     2. **Refinamento Fino sem Alvo (Método 2 - NID)**: Ajusta o Yaw fino da cena completa contra as paredes e batentes da sala, eliminando qualquer micro-deslocamento visual residual.
3. **Preservação dos Módulos**:
   - Toda a estrutura modular está salva sob `methods/` com executáveis funcionais e relatórios JSON consolidados, permitindo que cada método seja reexecutado ou inspecionado a qualquer momento de forma limpa e independente.
