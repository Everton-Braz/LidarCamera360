"""Forward FAST-LIVO2's console output without its per-scan noise.

The engine prints an 11-line timing table, feature counts and map-sliding debug lines for every scan:
~30,000 lines for a 2-minute walk, which made the workflow log megabytes long and slowed the log view.
The full output goes to a file beside the SLAM result; the console gets the rest plus a progress line.
"""
from __future__ import annotations

import re
import sys

ENGINE_LOG = 'fastlivo2.log'
SCAN_LINE = '[ LIO ] Update Voxel Map'
_ANSI = re.compile(r'\x1b\[[0-9;]*m')
_NOISE = re.compile(r'^(?:\[ LIO \] (?:Raw feature num|Update Voxel Map)'
                    r'|\[DEBUG\]: (?:Last sliding length|Map sliding using|Delete \d+ root voxels)'
                    r'|\+-{10,}\+$|\|\s)')
_AVERAGE = re.compile(r'^\|\s*Average Total Time\s*\|\s*([0-9.]+)')
REPORT_SHARE = 0.05
REPORT_EVERY_SCANS = 250  # when the scan count is unknown


def forward_engine_output(stream, log_path, total_scans=None, out=None):
    """Copy ``stream`` (bytes lines) to ``log_path`` and the useful part to ``out``. Returns the scan count."""
    out = out or sys.stdout
    scans, average, reported = 0, None, 0
    step = max(1, int(total_scans * REPORT_SHARE)) if total_scans else REPORT_EVERY_SCANS
    with open(log_path, 'w', encoding='utf-8', errors='replace') as log:
        for raw in stream:
            line = raw.decode('utf-8', errors='replace') if isinstance(raw, bytes) else raw
            log.write(line)
            text = _ANSI.sub('', line).rstrip()
            if text.startswith(SCAN_LINE):
                scans += 1
                if scans - reported >= step:
                    reported = scans
                    progress = (f'[{min(scans, total_scans)}/{total_scans}] SLAM scans mapped' if total_scans
                                else f'[*] FAST-LIVO2: {scans} scans mapped')
                    out.write(progress + (f', {average * 1000:.0f} ms per scan' if average else '') + '\n')
                    out.flush()
                continue
            match = _AVERAGE.match(text)
            if match:
                average = float(match.group(1))
                continue
            if not text or _NOISE.match(text):
                continue
            out.write(text + '\n')
            out.flush()
    if scans:
        out.write(f'[*] FAST-LIVO2 mapped {scans} scans' + (f' ({average * 1000:.0f} ms per scan)' if average else '')
                  + f'; full engine log: {log_path.name if hasattr(log_path, "name") else log_path}\n')
        out.flush()
    return scans
