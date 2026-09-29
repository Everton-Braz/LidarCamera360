"""Small helpers for keeping native command windows out of the GUI workflow."""
import os
import subprocess
import sys


def hidden_window_options():
    """Return Windows flags that suppress a child console/window at launch."""
    if os.name != 'nt':
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        'creationflags': subprocess.CREATE_NO_WINDOW,
        'startupinfo': startupinfo,
    }


def run_hidden_stream(command, *, env=None):
    """Run a native CLI invisibly while forwarding its output to the parent log."""
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
        env=env,
        **hidden_window_options(),
    ) as process:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        return process.wait()
