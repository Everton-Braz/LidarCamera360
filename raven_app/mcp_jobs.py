"""Background jobs for the MCP server: long CLI runs that report progress by id.

Workflow, colorization and rig calibration take minutes, while an MCP tool call
must answer promptly.  A job is the same headless CLI command the GUI runs, in its
own process with its own log; its state survives a server restart because it is
persisted next to the log.
"""
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

STAGE_RE = re.compile(r'^\[STAGE (\d+)/(\d+)\]\s*(.*)')
OUTPUT_RE = re.compile(r'(lidar_colored_[\w-]+\.(?:ply|pcd))')


def cli_command(args):
    """The headless CLI invocation for this installation (frozen exe or source checkout)."""
    if getattr(sys, 'frozen', False):
        return [sys.executable, '--headless'] + list(args)
    return [sys.executable, str(Path(__file__).resolve().parents[1] / 'raven.py'), '--headless'] + list(args)


class JobManager:
    def __init__(self, jobs_dir=None):
        if jobs_dir is None:
            from raven_app.config import get_config_dir
            jobs_dir = get_config_dir() / 'mcp_jobs'
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._processes = {}

    def _meta_path(self, job_id):
        return self.jobs_dir / f'{job_id}.json'

    def _write(self, meta):
        self._meta_path(meta['job_id']).write_text(json.dumps(meta, indent=2), encoding='utf-8')

    def _read(self, job_id):
        if not re.fullmatch(r'[0-9a-f]{12}', str(job_id)):
            raise ValueError(f'Invalid job id: {job_id}')
        path = self._meta_path(job_id)
        if not path.is_file():
            raise ValueError(f'Unknown job: {job_id}')
        return json.loads(path.read_text(encoding='utf-8'))

    def start(self, kind, args):
        job_id = uuid.uuid4().hex[:12]
        log_path = self.jobs_dir / f'{job_id}.log'
        command = cli_command(args)
        flags, startupinfo = 0, None
        if os.name == 'nt':
            flags = subprocess.CREATE_NEW_PROCESS_GROUP
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        log = log_path.open('w', encoding='utf-8')
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   creationflags=flags, startupinfo=startupinfo,
                                   env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'})
        log.close()
        self._processes[job_id] = process
        meta = {'job_id': job_id, 'kind': kind, 'args': list(args), 'pid': process.pid,
                'log': str(log_path), 'status': 'running',
                'started_at': datetime.now().astimezone().isoformat(timespec='seconds')}
        self._write(meta)
        return self.status(job_id)

    def _poll(self, meta):
        process = self._processes.get(meta['job_id'])
        if meta['status'] != 'running':
            return meta
        if process is None:
            # Started by an earlier server: only the persisted record is known.
            meta.update(status='unknown', note='started by an earlier MCP server session')
            return meta
        code = process.poll()
        if code is not None:
            meta.update(status='succeeded' if code == 0 else 'failed', exit_code=code,
                        finished_at=datetime.now().astimezone().isoformat(timespec='seconds'))
            if meta.get('cancel_requested'):
                meta['status'] = 'cancelled'
            self._write(meta)
        return meta

    def status(self, job_id, tail=40):
        meta = self._poll(self._read(job_id))
        lines = []
        log = Path(meta['log'])
        if log.is_file():
            lines = log.read_text(encoding='utf-8', errors='replace').splitlines()
        stage = next((m.groups() for m in map(STAGE_RE.match, reversed(lines)) if m), None)
        outputs = sorted({m.group(1) for line in lines for m in OUTPUT_RE.finditer(line)})
        verdicts = [line.strip() for line in lines if 'Photometric rig refinement' in line and
                    ('accepted' in line or 'rejected' in line)]
        return {**meta,
                'stage': {'index': int(stage[0]), 'total': int(stage[1]), 'name': stage[2]} if stage else None,
                'outputs': outputs, 'rig_refinement': verdicts[-4:], 'log_tail': lines[-tail:]}

    def cancel(self, job_id):
        meta = self._poll(self._read(job_id))
        if meta['status'] != 'running':
            return self.status(job_id)
        meta['cancel_requested'] = True
        self._write(meta)
        process = self._processes.get(job_id)
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(meta['pid']), '/T', '/F'], capture_output=True)
        elif process is not None:
            process.terminate()
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        return self.status(job_id)

    def list(self, limit=20):
        metas = sorted(self.jobs_dir.glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        return [{k: v for k, v in self._poll(json.loads(p.read_text(encoding='utf-8'))).items()
                 if k in ('job_id', 'kind', 'status', 'started_at', 'finished_at', 'exit_code')} for p in metas]

    def wait(self, job_id, timeout):
        """Block up to ``timeout`` seconds (tests and short jobs)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.status(job_id)
            if status['status'] != 'running':
                return status
            time.sleep(0.2)
        return self.status(job_id)
