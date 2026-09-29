"""Small, dependency-free HTML report for calibration review."""
import html
import json
from pathlib import Path

import numpy as np


def write_report(model, path):
    views = sorted(model['views'].items())
    width, height = 1000, 260
    lines = []
    for channel, color in enumerate(('#c33', '#298a42', '#386fc5')):
        values = [entry['log_gain'][channel] for _, entry in views]
        pts = ' '.join(f'{20 + i * 960/max(1,len(values)-1):.1f},{130 - float(v)*140:.1f}'
                       for i, v in enumerate(values))
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1"/>')
    rows = []
    for split in ('validation', 'test'):
        before, after = model['metrics'][split+'_before'], model['metrics'][split+'_after']
        for metric in ('linear_rmse', 'delta_e76_mean', 'delta_e76_p90'):
            rows.append(f'<tr><td>{split}</td><td>{metric}</td><td>{before[metric]:.5f}</td>'
                        f'<td>{after[metric]:.5f}</td></tr>')
    curves = []
    lenses = {}
    for name, entry in views:
        if entry['support'] >= 12:
            lenses.setdefault(name.split('/')[0], entry['vignette'])
    radius = np.linspace(0, 1, 100)
    for i, (name, coeff) in enumerate(lenses.items()):
        value = radius**2*coeff[0] + radius**4*coeff[1] + radius**6*coeff[2]
        pts = ' '.join(f'{20+r*960:.1f},{130-v*140:.1f}' for r, v in zip(radius, value))
        curves.append(f'<polyline points="{pts}" fill="none" stroke="{["#7759a3","#a26922"][i%2]}" stroke-width="2"/>')
    data = html.escape(json.dumps({k:v for k,v in model.items() if k != 'views'}, indent=2))
    body = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8">
<title>Validação fotométrica</title><style>body{{font:16px system-ui;max-width:1080px;margin:32px auto;padding:20px;color:#202633}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}svg{{width:100%;border:1px solid #ddd}}pre{{white-space:pre-wrap}}</style>
<h1>Compensação fotométrica: {html.escape(model['selected'])}</h1>
<p>60% dos pontos para ajuste, 20% para seleção e 20% para teste final. Métricas de correspondências SfM; não representam precisão geométrica ou inspeção visual da nuvem.</p>
<table><tr><th>Partição</th><th>Métrica</th><th>Antes</th><th>Depois</th></tr>{''.join(rows)}</table>
<h2>Log-ganhos RGB por imagem</h2><p>Ordenação por nome; linha central = identidade. Imagens sem suporte mantêm identidade.</p>
<svg viewBox="0 0 {width} {height}"><path d="M20 130H980" stroke="#bbb"/>{''.join(lines)}</svg>
<h2>Vinheta por lente</h2><p>Raio normalizado: 0 a 1; eixo vertical: log da resposta estimada, com zero na linha central.</p>
<svg viewBox="0 0 {width} {height}"><path d="M20 130H980" stroke="#bbb"/>{''.join(curves)}</svg>
<details><summary>Dados e tempos</summary><pre>{data}</pre></details></html>'''
    Path(path).write_text(body, encoding='utf-8')
