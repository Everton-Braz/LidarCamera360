"""Asynchronous desktop launcher for the headless RF-DETR resource installer."""
import json
from pathlib import Path
import sys

from PyQt6.QtCore import QObject, QProcess, pyqtSignal


class MaskResourceDownload(QObject):
    progress = pyqtSignal(str, int)
    completed = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = None
        self._output = []
        self._pending = b''

    def start(self):
        if self._process is not None:
            return False
        process = QProcess(self)
        self._process = process
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_output)
        process.errorOccurred.connect(self._process_error)
        process.finished.connect(self._finished)
        if getattr(sys, 'frozen', False):
            process.setProgram(sys.executable)
            process.setArguments(['--headless', 'download-mask-resources'])
        else:
            entry = Path(__file__).resolve().parents[1] / 'raven.py'
            process.setProgram(sys.executable)
            process.setArguments([str(entry), '--headless', 'download-mask-resources'])
            process.setWorkingDirectory(str(entry.parent))
        process.start()
        return True

    def cancel(self):
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            self._process.terminate()
            if not self._process.waitForFinished(2000):
                self._process.kill()

    def _read_output(self):
        process = self._process
        if process is None:
            return
        self._pending += bytes(process.readAllStandardOutput())
        while b'\n' in self._pending:
            line, self._pending = self._pending.split(b'\n', 1)
            self._consume_line(line.decode('utf-8', errors='replace').strip())

    def _consume_line(self, line):
        if not line:
            return
        try:
            event = json.loads(line)
        except ValueError:
            self._output.append(line)
            self._output = self._output[-30:]
            return
        if event.get('type') == 'resource-progress':
            self.progress.emit(str(event.get('resource', 'resources')),
                               max(0, min(100, int(event.get('percent', 0)))))
        else:
            self._output.append(line)
            self._output = self._output[-30:]

    def _finished(self, code, _status):
        if self._process is None:
            return
        self._read_output()
        if self._pending.strip():
            self._consume_line(self._pending.decode('utf-8', errors='replace').strip())
        self._pending = b''
        process, self._process = self._process, None
        if process is not None:
            process.deleteLater()
        self.completed.emit(int(code), '\n'.join(self._output[-12:]))

    def _process_error(self, error):
        if error != QProcess.ProcessError.FailedToStart or self._process is None:
            return
        process, self._process = self._process, None
        message = process.errorString()
        process.deleteLater()
        self.completed.emit(2, message)
