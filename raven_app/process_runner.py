"""Background subprocess runner using QThread for Fluent UI integration."""
import os
from pathlib import Path
import signal
import subprocess
import sys
from PyQt6.QtCore import QThread, pyqtSignal


class ProcessRunner(QThread):
    """Executes RavenCalibrator CLI commands in a background thread and emits signals."""
    log_received = pyqtSignal(str)
    started = pyqtSignal()
    finished = pyqtSignal(int)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._args = []
        self._process = None
        self._is_cancelling = False

    def start_job(self, args: list[str]):
        """Start a new CLI execution with given arguments."""
        if self.isRunning():
            return False
        self._args = args
        self._is_cancelling = False
        self.start()
        return True

    def cancel(self):
        """Send graceful cancellation signal (CTRL_BREAK_EVENT on Windows to flush maps)."""
        if self._process and self._process.poll() is None:
            self._is_cancelling = True
            try:
                if os.name == 'nt':
                    self._process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    self._process.send_signal(signal.SIGINT)
                self.log_received.emit("\n[!] Cancellation requested: saving and flushing accumulated data...\n")
            except OSError as e:
                self.log_received.emit(f"\n[!] Cancellation signal failed: {e}\n")
                try:
                    self._process.terminate()
                except Exception:
                    pass

    @property
    def is_busy(self) -> bool:
        return self.isRunning()

    def run(self):
        self.started.emit()
        try:
            cmd = (
                [sys.executable]
                if getattr(sys, 'frozen', False)
                else [sys.executable, str(Path(__file__).resolve().parents[1] / 'raven.py')]
            ) + ['--headless'] + self._args

            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                creationflags=flags
            )

            for line in self._process.stdout:
                self.log_received.emit(line)

            exit_code = self._process.wait()
            self.finished.emit(exit_code)
        except Exception as e:
            self.error_occurred.emit(str(e))
            self.finished.emit(2)
        finally:
            self._process = None
