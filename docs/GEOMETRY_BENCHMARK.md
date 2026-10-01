# Validação de geometria Home3 — 30/09/2026

O comparador `scripts/benchmark_geometry.py` mede duas propriedades distintas:

- **Precisão local de superfície:** RMSE ponto-plano em pontos reservados, nas mesmas 200 regiões espaciais. Não é erro contra um levantamento de referência.
- **Concordância de trajetória:** validação cruzada em cinco blocos temporais das posições de câmeras SfM e SLAM, com vizinhos temporais removidos do treino. Não mede diretamente precisão das paredes.

## Protocolo

Os ensaios usam integralmente `LIDAR_20260927Home3.bag`, desde a inicialização imóvel, com o mesmo executável nativo e quatro threads. Cada ensaio grava configuração, log, trajetória e nuvem em um diretório novo. Os ensaios são LIO: este bag contém LiDAR e IMU, sem imagens da câmera frontal.

O comparador amostra dois milhões de pontos com semente fixa e os divide em treino e avaliação. Escolhe regiões planas de raio 35 cm usando apenas o treino da referência, com centros espacialmente separados. Em cada mapa, ajusta cada plano nos pontos de treino e mede todos os resíduos dos pontos de avaliação daquele volume, sem corte por resíduo. As regiões têm o mesmo peso no RMSE agregado. O alinhamento entre mapas é rígido, estimado pelas trajetórias, sem ajuste de escala.

Uma configuração candidata só é recomendada se reduzir o RMSE local em pelo menos 3%, sem piorar o P90 das regiões ou a concordância de trajetória. Também precisa manter pelo menos 95% das regiões, normais compatíveis e 99% da contagem total de pontos. Essa recomendação é seleção por validação neste dataset, não demonstra generalização para outros levantamentos.

## Reprodução

```powershell
python scripts/benchmark_geometry.py --dataset scratch/Home3 --reference-slam scratch/geometry_20260930/baseline/output --candidate-slam scratch/geometry_20260930/finer/output scratch/geometry_20260930/coarser/output scratch/geometry_20260930/strict_planes/output --output scratch/geometry_20260930/comparison_final.json
python -m unittest discover -s tests -p test_geometry_quality.py -v
```

Os testes sintéticos verificam escala rígida, sensibilidade ao ruído, rejeição de ganhos obtidos perdendo regiões e inclusão de resíduos altos na avaliação.

## Resultados

| Configuração: filtro / voxel (m) | RMSE local (mm) | CV de trajetória (cm) | Decisão |
|---|---:|---:|---|
| Atual: 0,03 / 0,15 | 10,300 | 11,461 | Referência |
| Fina: 0,02 / 0,10 | 10,430 | 11,869 | Rejeitada |
| Maior: 0,03 / 0,20 | 13,272 | 15,999 | Rejeitada |
| Atual com limiar planar 0,0005 | 10,371 | 11,645 | Rejeitada |

Todos produziram 30.439.179 pontos e preservaram as 200 regiões avaliadas. A repetição da configuração atual reproduziu as métricas do mapa original. Os tempos nativos foram 94,57 / 129,38 / 115,01 segundos; havia processamento fotométrico concorrente, portanto não são um benchmark isolado de desempenho.

O quarto ensaio alterou apenas `lio.min_eigen_value` de 0,002 para 0,0005 e levou 127,70 segundos no motor. Esse parâmetro limita o menor autovalor da covariância para aceitar uma região como plano. O critério mais rigoroso melhorou ligeiramente as 114 regiões verticais (9,590 → 9,569 mm), mas piorou as 86 horizontais (11,171 → 11,347 mm) e a trajetória; por isso foi rejeitado.

Nenhum candidato atingiu o critério de promoção. A configuração de produção permanece intacta. O resultado desta rodada é uma medição reproduzível e a rejeição de regressões, não uma alegação de nova melhora do XYZ. Os dados detalhados estão em `scratch/geometry_20260930/comparison_final.json`.

As correções PPISP/grade bilateral atuam sobre RGB. Elas não alteram XYZ e não podem ser creditadas por uma redução de RMSE geométrico.
