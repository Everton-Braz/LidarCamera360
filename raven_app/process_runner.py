"""Background subprocess runner with live UI output and persistent run logs."""
from datetime import datetime
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading

from PyQt6.QtCore import QThread, pyqtSignal


def make_run_log_path(root: str | Path, label: str, started_at=None) -> Path:
    """Return a unique per-run log path below ``root/logs``."""
    stamp = (started_at or datetime.now()).strftime('%Y%m%d_%H%M%S_%f')
    safe_label = re.sub(r'[^A-Za-z0-9._-]+', '_', str(label)).strip('._-') or 'workflow'
    return Path(root) / 'logs' / f'{safe_label}_{stamp}.log'


def _infer_log_target(args: list[str]) -> tuple[Path, str]:
    """Choose the workflow's output directory and a useful automatic log name."""
    command = args[0] if args else 'workflow'
    root = None
    for option in ('--output', '--dataset', '--out', '--workspace'):
        try:
            root = Path(args[args.index(option) + 1]).expanduser()
            break
        except (ValueError, IndexError):
            continue
    if root is None:
        from raven_app.config import get_config_dir
        root = get_config_dir() / 'run-logs'
    label = f'{root.name}_{command}'
    return root, label


class ProcessRunner(QThread):
    """Execute RavenCalibrator CLI commands without opening console windows."""
    log_received = pyqtSignal(str)
    started = pyqtSignal()
    finished = pyqtSignal(int)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._args = []
        self._process = None
        self._is_cancelling = False
        self._log_path = None
        self._initial_log = ''
        self._log_stream = None
        self._log_lock = threading.Lock()

    @property
    def log_path(self) -> Path | None:
        """Path of the current or most recent automatic run log."""
        return self._log_path

    def start_job(self, args: list[str], log_path: str | Path | None = None,
                  initial_log: str = ''):
        """Start a CLI execution and persist its streamed output to a unique log."""
        if self.isRunning():
            return False
        self._args = list(args)
        self._is_cancelling = False
        if log_path is None:
            root, label = _infer_log_target(self._args)
            self._log_path = make_run_log_path(root, label)
        else:
            self._log_path = Path(log_path)
        self._initial_log = initial_log
        self.start()
        return True

    def _emit_log(self, text: str):
        """Write before emitting so a visible line is already safe on disk."""
        if not text:
            return
        with self._log_lock:
            if self._log_stream is not None:
                self._log_stream.write(text)
                self._log_stream.flush()
        self.log_received.emit(text)

    def cancel(self):
        """Request graceful cancellation so the CLI can flush accumulated data."""
        if self._process and self._process.poll() is None:
            self._is_cancelling = True
            try:
                if os.name == 'nt':
                    self._process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self._process.send_signal(signal.SIGINT)
                self._emit_log("\n[!] Cancellation requested: saving and flushing accumulated data...\n")
            except OSError as exc:
                self._emit_log(f"\n[!] Cancellation signal failed: {exc}\n")
                try:
                    self._process.terminate()
                except Exception:
                    pass

    @property
    def is_busy(self) -> bool:
        return self.isRunning()

    def run(self):
        self.started.emit()
        exit_code = 2
        try:
            cmd = (
                [sys.executable]
                if getattr(sys, 'frozen', False)
                else [sys.executable, str(Path(__file__).resolve().parents[1] / 'raven.py')]
            ) + ['--headless'] + self._args

            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_stream = self._log_path.open('w', encoding='utf-8', buffering=1)
            self._emit_log(
                f"=== {self._log_path.stem} ===\n"
                f"Started: {datetime.now().astimezone().isoformat(timespec='seconds')}\n"
                f"Command: {subprocess.list2cmdline(cmd) if os.name == 'nt' else shlex.join(cmd)}\n\n"
            )
            if self._initial_log:
                self._emit_log(self._initial_log.rstrip() + '\n\n')

            flags = 0
            startupinfo = None
            if os.name == 'nt':
                # Hide the console window while retaining a console process group
                # so CTRL_BREAK_EVENT can still request a graceful pipeline stop.
                flags = subprocess.CREATE_NEW_PROCESS_GROUP
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                creationflags=flags,
                startupinfo=startupinfo
            )

            for line in self._process.stdout:
                self._emit_log(line)

            exit_code = self._process.wait()
        except Exception as exc:
            self._emit_log(f"\n[ERROR] {type(exc).__name__}: {exc}\n")
            self.error_occurred.emit(str(exc))
            exit_code = 2
        finally:
            self._emit_log(
                f"\n=== Finished: {datetime.now().astimezone().isoformat(timespec='seconds')} "
                f"(exit code {exit_code}) ===\n"
            )
            with self._log_lock:
                if self._log_stream is not None:
                    self._log_stream.close()
                    self._log_stream = None
            self._process = None
            self.finished.emit(exit_code)
