# Compensação fotométrica: diagnóstico e proposta

Data: 2026-09-29. Escopo: implementação log-linear nativa no caminho Vulkan e validação no Home3. O texto anexado foi avaliado como proposta; PPISP/bilateral grid completos continuam fora do modo padrão.

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

Vulkan: `raven_app/vulkan_engine.py::colorize_views` transmite caminhos e câmeras pelo protocolo RVC1/RVC2. `colorize_consensus.comp` amostra RGB e armazena RGB8; `resolve_consensus.comp` resolve as três observações. Não há normalização exposição/WB/vinheta. O consenso opera nos valores codificados da imagem, sem conversão explícita para luz linear.

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

Conclusão: começar por compensação robusta de ganhos/WB e vinheta opcional usando as dependências existentes. Não há evidência nesta análise de que PPISP ou bilateral grid sejam necessários, nem de que removam todas as manchas apresentadas.
