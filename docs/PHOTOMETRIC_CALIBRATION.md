# Compensação fotométrica e calibração geométrica

Data: 2026-09-29. Escopo: compensação log-linear, variante compacta PPISP e grade bilateral no caminho Vulkan, com validação no Home3. O texto anexado foi avaliado como proposta técnica.

## Evidência local

- Resultado: `lidar_colored_sfm_consensus_20260927_194743.ply`, 454.772.027 bytes, 30.318.123 pontos. Logs confirmam Vulkan e 3.556 vistas. A etapa do pipeline que inclui colorização e remoção do operador levou 425,1 s; não é tempo isolado de GPU. Pipeline completo: 2.002,8 s.
- `colmap_to_lidar_alignment.json`: captura timelapse, RMSE de trajetória 11,653 cm. Isso não mede diretamente erro de reprojeção nem precisão das paredes, mas exige diagnóstico geométrico junto do fotométrico.
- Há máscaras de pessoas; o log registra remoção de 121.056 pontos e proteção de 44.723 pontos planares. Não ampliar remoção geométrica para tratar manchas de cor.
- A imagem fornecida mostra variações de brilho/cor, mas sozinha não permite separar exposição, sombras reais, oclusões, poses e visualização de pontos.

## Execução real antes/depois

Os arquivos foram gerados em `scratch/photometric_home3/full_comparison/` a partir da mesma nuvem XYZ e da mesma ordem de pontos:

- `before.ply`: 454.772.027 bytes, 30.318.123 pontos, 197,7 s de Vulkan.
- `after.ply`: 454.772.027 bytes, 30.318.123 pontos, 208,4 s de Vulkan.
- XYZ idêntico; 25.547.851 pontos tiveram alguma mudança RGB. O acréscimo medido foi 5,4% no tempo de colorização e não houve aumento do PLY.
- Média de distância RGB entre os arquivos: 25,726 níveis; isso mede a intensidade da correção aplicada, não qualidade por si só.
- Hold-out SfM: ΔE76 médio 5,0069 → 4,1072; RMSE linear 0,05913 → 0,04746; P90 de ΔE 10,61 → 8,17.

O ajuste recalculado dentro da cópia Home3 selecionou `gains` por imagem. A vinheta foi testada, mas não passou o ganho adicional de validação depois de limitar a aplicação às vistas com suporte suficiente. Isso evita sobrecorrigir regiões pouco observadas.

## Fluxo confirmado no código atual

`scripts/colorize_sfm_spirula_method.py` é um wrapper. A implementação está em `scripts/pipeline_auto_calibrator_and_colorizer.py::colorize_via_spirula_sfm` e `colorize_via_direct_rigid`. As vistas contêm caminho, pose e intrínsecos; `cam_id` identifica câmera no COLMAP e os arquivos seguem `cam0/` e `cam1/`.

CPU: projeção thin-prism → círculo útil → z-buffer → máscara → peso radial/distância → amostragem bilinear → top-3 → rejeição pela distância RGB de 45 → média ponderada. O peso radial não é uma normal de superfície; não confundir com incidência rasante. A seleção de quadros calcula nitidez em `video.py`, mas o peso por observação aqui não inclui nitidez local.

Vulkan: `raven_app/vulkan_engine.py::colorize_views` transmite caminhos e câmeras por RVC1/RVC2; RVC3/RVC4 acrescentam ganhos/vinheta; RVC5/RVC6 acrescentam homografia PPISP e grade bilateral. A correção opera em RGB linear antes de armazenar as observações RGB8. `resolve_consensus.comp` mantém o consenso existente em RGB codificado.

As imagens deste conjunto são JPEG. O caminho CPU decodifica BGR8 e converte para RGB8. Tratar sRGB como hipótese operacional configurável; JPEG não comprova a curva de resposta real do ISP. O parser `parse_insv_telemetry.py` identifica registros Exposure/ExposureSecondary, mas não fornece aqui um prior fotométrico validado de ISO × obturador × ganho. `video.py` usa os primeiros oito bytes de Exposure como origem temporal. Não interpretar esse timestamp como duração de exposição.

## Implementação recomendada, com custo limitado

1. Coletar observações dos mesmos pontos em diferentes vistas, usando projeção, máscaras e oclusão existentes. Começar com 30–50 mil pontos espacialmente distribuídos, ≥3 vistas e teto explícito de observações; subir para 300 mil apenas se a cobertura exigir. Não construir z-buffer apenas com a amostra esparsa: isso produz falsas visibilidades. Reutilizar profundidade calculada da nuvem completa ou representação conservadora de oclusores.
2. Reservar 20% dos IDs de pontos antes do ajuste. Separar também regiões/tempos para uma avaliação mais exigente. Rejeitar saturação, sombras extremas, bordas de oclusão, máscaras e observações inconsistentes geometricamente. Guardar point_id, image_id, lens_id, u, v, RGB e peso.
3. Ajustar NumPy/SciPy com Huber/IRLS em log de luz linear: radiância por ponto + ganho RGB por imagem + vinheta radial compartilhada por lente. NumPy/SciPy já constam de requirements.txt. Eliminar/alternar radiância latente para limitar memória; evitar matrizes densas ponto × quadro.
4. Estimar primeiro ganhos RGB. Exposição e WB separados têm ambiguidade: definir exposição como média dos três log-ganhos e WB como resíduo de soma zero por imagem. Ancorar brilho por componente conexa do grafo de sobreposição. Não impor normalização independente das lentes que impeça corrigir uma diferença real entre elas.
5. Só habilitar vinheta quando houver cobertura radial e validação suficiente; regularizar fortemente e limitar ganho. Preservar identidade em imagens sem suporte e registrar a confiança. Usar centro óptico e dimensões efetivos do COLMAP, não constantes universais.
6. Protótipo: pasta alternativa preservando nomes, dimensões, timestamps e correspondência das máscaras. Não substituir imagens usadas no SfM. Cópias normalizadas permitem validar o Vulkan atual, mas acrescentam decodificação, escrita e espaço; JPEG acrescenta recompressão.
7. Produção: aplicar parâmetros na imagem já decodificada antes do upload ou diretamente na amostra, antes do consenso, com caminho CPU equivalente. Isso evita regravar milhares de imagens. Manter precisão suficiente antes da decisão de consenso; avaliar média em luz linear separadamente da mudança do limiar de rejeição.
8. Cache versionado por imagens, poses, intrínsecos, máscaras e configuração. Opção desligada preserva resultado existente. Ativação automática somente após critérios de validação; nunca ativar ajustes sem suporte.

## PPISP e bilateral grid

[PPISP oficial](https://github.com/nv-tlabs/ppisp) modela exposição por quadro, vinheta por câmera, correção cromática e CRF. A implementação usa torch e extensão CUDA; seu uso documentado é um modelo direto radiância → imagem em treinamento. Para normalizar fotos é preciso estabelecer referência e validar a inversão; `frame_idx=-1` não é uma API de inversão. Controller pode ser desativado com `controller_distillation=False`. Bom candidato experimental se a base leve deixar resíduos sistemáticos de resposta tonal; não incluir dependências no executável padrão.

[fused-bilagrid oficial](https://github.com/harry7557558/fused-bilagrid) aceita coordenadas irregulares, mas o README documenta teste Ubuntu 22.04/PyTorch 2.1.2+cu118; não estabelece compatibilidade Windows. Seus ganhos publicados são de treinamento de splats, não do nosso colorizador. Uma grade espacial/intensidade pode corrigir resíduos locais, mas também absorver sombras, textura e erro de alinhamento. Aplicar somente depois da base, com identidade, TV e penalização de magnitude; desligada por padrão.

Uma grade afim RGB 8×8×4 com 12 coeficientes float32 tem 12.288 bytes por imagem: para 3.556 imagens, aproximadamente 41,7 MiB apenas de parâmetros, sem gradientes/otimizador. Ganhos RGB por imagem ocupam aproximadamente 41,7 KiB em float32. O problema de distribuição das versões oficiais está também nos runtimes torch/CUDA, não apenas nos coeficientes. Não estimar aumento exato do bundle sem empacotar. Inferência futura de grade em Vulkan não exige necessariamente distribuir o runtime de treinamento.

## Critérios para decidir e entregar

- Comparar pontos reservados: dispersão RGB linear, ΔE em Lab sob referência declarada, percentis e taxa de observações válidas. Comparar fronteiras de melhor vista só dentro de superfícies coerentes para não penalizar bordas reais.
- Mostrar recortes idênticos antes/depois e mapas de confiança; avaliar fachada, teto, sombras e bordas. Exposição não corrige geometria nem recupera canais saturados.
- Medir separadamente coleta, ajuste, aplicação, colorização, RAM/VRAM, cache e tamanho do pacote. Meta proposta: acréscimo ≤10% no tempo total de colorização após integração; é um orçamento de engenharia, não resultado medido.
- Testes de recuperação sintética, identidade, grafo desconexo, baixa cobertura, saturação, cache invalidado, JSON e paridade CPU/Vulkan. Smoke test do executável exato em CWD/PATH isolados depois da integração.
- Comparação real deve escrever novos arquivos; manter originais. Só promover a correção se melhorar hold-out e inspeção visual sem regressões relevantes.

## Modo PPISP + bilateral grid

`--photometric ppisp-bilateral` usa exposição/WB log-linear, homografia cromática PPISP no espaço RGI (R, G, soma RGB) preservando intensidade, seguida de grade trilinear XY/luminância com ganhos RGB. Trata-se de uma variante compacta sem CRF aprendida: não é paridade completa da implementação NVIDIA nem da grade afim de 12 coeficientes do Spirula. A correção observada → referência é ajustada diretamente usando correspondências da mesma cena.

O contrato CPU/Vulkan usa homografia 3×3 row-major e grade `[4,3,3,3]` em ordem intensidade/Y/X/RGB, coordenadas espaciais `u/3840`, `v/3840`, guia Rec.709 do RGB linear após a homografia. A grade aplica `exp(clamp(log_gain, -0.35, 0.35))`. São 117 floats adicionais por vista (468 bytes), sem novas bibliotecas de inferência. Os JSONs usam schema 2; schema 1 continua aceito pelo modo log-linear. A grade corrige amostras fotométricas; não suaviza XYZ ou cores entre superfícies da nuvem.

O ajuste e a seleção usam somente as partições de treino e validação. O teste reservado serve para o relatório, sem participar da seleção. Componentes que não melhoram a validação retornam à identidade. O modo permanece selecionável e os parâmetros ficam em cache, sem regravar JPEGs.

## Resultado Home3 da implementação

O ajuste avançado foi executado sobre 36.266 pontos SfM e 182.113 observações. No conjunto de validação, o RMSE linear caiu de `0,048005` para `0,045835` (−4,52%) e o ΔE76 médio de `4,1307` para `3,9911` (−3,38%). No teste reservado, o RMSE caiu de `0,047462` para `0,045480` (−4,18%) e o ΔE76 médio de `4,1072` para `3,9712` (−3,31%). O P90 de ΔE caiu de `8,1663` para `7,9024`.

O arquivo aplicado à nuvem é [`advanced_params.json`](../scratch/photometric_home3/advanced/advanced_params.json). A nuvem gerada é [`after.ply`](../scratch/photometric_home3/advanced_geometry/after.ply), com 454.772.027 bytes e 30.318.123 pontos. O XYZ é idêntico ao arquivo de entrada; a mudança é de pose das vistas de câmera e de RGB. A colorização avançada levou `178,21 s` no Vulkan, contra `149,12 s` para o modo log-linear no mesmo executável e nuvem, acréscimo de 19,5% nessa medição. O custo é dominado pela passagem de 3.556 imagens e I/O; o modelo em si é pequeno.

O refinamento geométrico está documentado em [`geometry_refinement_validation.json`](../scratch/Home3/geometry_v2/geometry_refinement_validation.json): RMSE bloqueado em poses de teste `13,273 → 11,461 cm` (−13,66%), quatro de cinco blocos melhores, mesmo conjunto temporal válido e sem corte por resíduo no teste. O candidato de alinhamento e o sidecar de rig são usados somente na nuvem avançada; o dataset original permanece intacto.

O executável Vulkan passou de 118.784 para 121.344 bytes (+2.560 bytes); o shader de consenso passou de 18.292 para 24.728 bytes (+6.436 bytes). Isso é a medição do componente nativo, não uma promessa de tamanho final do instalador portátil, que precisa ser reempacotado para medir todas as dependências.

O empacotamento isolado atual produziu [`LidarCamera360.exe`](../scratch/package_ppisp/single-file/LidarCamera360.exe) com 265.178.242 bytes. O smoke test executado a partir de CWD e `PATH` isolados passou (doctor, GUI, entrada inválida e colorização direta). O PPISP não adicionou torch ou outro runtime: o modo usa NumPy/SciPy já presentes. O tamanho do executável portátil deve ser comparado com um pacote gerado pelo mesmo checkout e pelas mesmas opções; o número anterior disponível no histórico foi produzido antes das alterações independentes de terceira câmera.

Comandos:

```powershell
python lidarcamera360.py --headless photometric-fit --dataset scratch/Home3 --masks-dir scratch/Home3/masks --mode ppisp-bilateral
python lidarcamera360.py --headless colorize --dataset scratch/Home3 --method sfm --masks-dir scratch/Home3/masks --photometric ppisp-bilateral
```

PPISP não corrige poses erradas, oclusões nem canais saturados. O refinamento de alinhamento muda as poses das câmeras; não reconstrói o SLAM nem altera o XYZ exportado. Os resultados quantitativos da nova execução são registrados junto aos arquivos de comparação.
