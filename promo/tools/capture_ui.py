"""Capture the real GUI pages as PNGs for release media.

Usage: python promo/tools/capture_ui.py <out_dir> [--theme dark|light] [--size 1600x1000]
Runs the actual RavenMainWindow (no mockups) and grabs each navigation page.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--theme', default='dark', choices=('dark', 'light'))
    ap.add_argument('--size', default='1600x1000')
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w, h = (int(v) for v in args.size.split('x'))

    app = QApplication.instance() or QApplication(sys.argv)
    from raven_app.i18n import set_language
    set_language('en')
    from raven_app.gui import RavenMainWindow
    from qfluentwidgets import setTheme, Theme

    win = RavenMainWindow()
    setTheme(Theme.DARK if args.theme == 'dark' else Theme.LIGHT)
    if hasattr(win, 'setMicaEffectEnabled'):
        win.setMicaEffectEnabled(False)
    win.resize(w, h)
    win.move(20, 20)
    win.show()
    win.raise_()
    win.activateWindow()

    pages = [
        ('01_unified_studio', win.workflow_view),
        ('02_slam', win.slam_view),
        ('03_colorize', win.colorize_view),
        ('04_inspector', win.inspect_view),
        ('05_rig_profiles', win.calib_view),
        ('06_doctor', win.doctor_view),
        ('07_settings', win.settings_view),
    ]
    state = {'i': 0}

    def step():
        if state['i'] > 0:
            name, _ = pages[state['i'] - 1]
            path = out / f'{args.theme}_{name}.png'
            win.screen().grabWindow(int(win.winId())).save(str(path))
            print(path)
        if state['i'] >= len(pages):
            win.close()
            app.quit()
            return
        win.switchTo(pages[state['i']][1])
        state['i'] += 1
        QTimer.singleShot(900, step)

    QTimer.singleShot(1500, step)
    app.exec()


if __name__ == '__main__':
    main()
