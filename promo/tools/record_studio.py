"""Record the real Unified Studio running a workflow, for release media.

Usage:
  python promo/tools/record_studio.py <out_dir> [--size 1600x900] [--live-sec 90] -- workflow --bag ... (a Studio command line)

Fills the Studio from a workflow command line (the "Command:" line of a run log, after
`--headless`), scrolls through the settings, presses Start and records the window:
every 100 ms for the setup and the first --live-sec of the run, then once a second until
the run ends. Frames go to <out_dir>/frames; studio_live.mp4 (real time) and
studio_timelapse.mp4 (1 s per frame -> 30x) and
studio_end.mp4 (the last 10 s) are assembled with ffmpeg at the end.
Run it with an interpreter that has both PyQt6 and the pipeline deps (build/portable-env).
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PyQt6.QtCore import QPoint, Qt, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

LIVE_MS, SLOW_MS = 100, 1000


def parse_workflow(argv):
    """The flags the Studio writes, as a dict (flag -> value or True)."""
    if argv and argv[0] == "workflow":
        argv = argv[1:]
    out, i = {}, 0
    while i < len(argv):
        flag = argv[i]
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[flag] = argv[i + 1]
            i += 2
        else:
            out[flag] = True
            i += 1
    return out


def pick(combo, want):
    """Select the combo item whose data or text matches want (a profile matches by file name)."""
    for k in range(combo.count()):
        data, text = combo.itemData(k), combo.itemText(k)
        if want in (data, text) or (data and Path(str(data)).name == Path(str(want)).name):
            combo.setCurrentIndex(k)
            return True
    raise SystemExit(f"no item {want!r} in {combo.objectName() or type(combo).__name__}")


def fill(view, flags):
    v = view
    v.bag_input.setText(flags["--bag"])
    v.out_input.setText(flags["--output"])
    v.insv_input.setText(flags.get("--insv", ""))
    if "--secondary-insv" in flags:
        v.dual_insv_chk.setChecked(True)
        v.secondary_insv_input.setText(flags["--secondary-insv"])
    v.scanner_combo.setCurrentIndex(["auto", "eagle", "raven"].index(flags.get("--scanner", "auto")))
    v.method_combo.setCurrentIndex(["sfm", "direct", "all"].index(flags.get("--method", "sfm")))
    v.lidar_topic.setText(flags.get("--lidar-topic", ""))
    v.imu_topic.setText(flags.get("--imu-topic", ""))
    v.threads_spin.setValue(int(flags.get("--threads", 4)))
    v.fps_spin.setValue(float(flags.get("--fps", 1.0)))
    pick(v.photometric_combo, flags.get("--photometric", "off"))
    v.recalibrate_chk.setChecked("--recalibrate" in flags)
    pick(v.rig_refine_combo, flags.get("--rig-refine", "off"))
    v.time_window_spin.setValue(float(flags.get("--time-window", 6)))
    v.remove_dynamic_chk.setChecked("--remove-dynamic" in flags)
    if "--profile" in flags:
        pick(v.rig_profile_combo, flags["--profile"])
    v.trim_chk.setChecked("--trim-start" in flags or "--trim-end-at" in flags)
    v.trim_start_spin.setValue(float(flags.get("--trim-start", 0)))
    v.trim_end_spin.setValue(float(flags.get("--trim-end-at", 0)))
    v.mask_persons_chk.setChecked("--mask-persons" in flags)
    if "--mask-persons" in flags:
        pick(v.mask_backend_combo, flags.get("--mask-backend", "tensorrt"))
        if "--mask-config" in flags:
            v.mask_config = json.loads(Path(flags["--mask-config"]).read_text(encoding="utf-8"))
        v.operator_radius_spin.setValue(float(flags.get("--operator-radius", 1.0)))
    v.lio_switch.setChecked("--lio" in flags)
    v.revisit_refinement_chk.setChecked("--revisit-refinement" in flags)
    if "--dt" in flags:
        v.auto_sync_chk.setChecked(False)
        v.dt_input.setText(flags["--dt"])
    v.chk_ply.setChecked("--export-ply" in flags)
    v.chk_pcd.setChecked("--export-pcd" in flags)
    v.chk_colmap.setChecked("--export-colmap" in flags)
    if "--seed-percent" in flags:
        v.seed_percent_spin.setValue(int(flags["--seed-percent"]))
    v.process_gps_chk.setChecked("--process-gps" in flags)


def assemble(frames, out_dir, size):
    """studio_live.mp4 from the 100 ms frames, studio_timelapse.mp4 from the 1 s frames."""
    w, h = size
    scale = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
    for kind, rate in (("live", 10), ("timelapse", 30), ("end", 10)):
        listing = frames / f"{kind}.txt"
        names = sorted(p.name for p in frames.glob(f"*_{kind}.jpg"))
        if not names:
            continue
        listing.write_text("".join(f"file '{n}'\nduration {1 / rate}\n" for n in names), encoding="utf-8")
        dst = out_dir / f"studio_{kind}.mp4"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(listing), "-vf", scale, "-r", "30", "-c:v", "libx264", "-crf", "18",
                        "-preset", "slow", str(dst)], check=True)
        print(dst)


def main():
    argv = sys.argv[1:]
    cmd = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--size", default="1600x900", help="window size in logical pixels")
    ap.add_argument("--video-size", default="1920x1080")
    ap.add_argument("--live-sec", type=float, default=90.0, help="seconds of the run recorded in real time")
    ap.add_argument("--dry-run", action="store_true", help="fill, compare the command line and quit")
    args = ap.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    flags = parse_workflow(cmd)
    out_dir = Path(args.out_dir)
    frames = out_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    w, h = (int(x) for x in args.size.split("x"))
    vw = int(args.video_size.split("x")[0])

    app = QApplication.instance() or QApplication(sys.argv)
    from raven_app.i18n import set_language
    set_language("en")
    from qfluentwidgets import SmoothScrollArea, Theme, setTheme
    from raven_app.gui import RavenMainWindow

    win = RavenMainWindow()
    setTheme(Theme.DARK)
    if hasattr(win, "setMicaEffectEnabled"):
        win.setMicaEffectEnabled(False)
    win.resize(w, h)
    win.show()
    win.switchTo(win.workflow_view)
    view = win.workflow_view
    fill(view, flags)
    scroll = view.findChild(SmoothScrollArea)
    bar = scroll.verticalScrollBar()

    sent = {}
    real_start = view.runner.start_job

    def start_job(job_args, **kw):
        sent["args"] = list(job_args)
        if args.dry_run:
            return False
        return real_start(job_args, **kw)

    view.runner.start_job = start_job
    t0 = time.monotonic()
    st = {"phase": "setup", "run_t": None, "done_t": None}

    def grab(kind):
        ms = int((time.monotonic() - t0) * 1000)
        shot = win.grab()
        if shot.width() > vw:
            shot = shot.scaledToWidth(vw, Qt.TransformationMode.SmoothTransformation)
        shot.save(str(frames / f"{ms:08d}_{kind}.jpg"), "JPG", 92)

    def finished(*_):
        st["done_t"] = time.monotonic()

    view.runner.finished.connect(finished)

    def tick():
        now = time.monotonic() - t0
        if st["phase"] == "setup":
            # 3 s still on the inputs, 12 s scrolling down to the Start button, 1 s pause, Start.
            if now > 3.0:
                frac = min(1.0, (now - 3.0) / 12.0)
                bar.setValue(int(bar.maximum() * (0.5 - 0.5 * math.cos(frac * math.pi))))
            grab("live")
            if now > 16.0:
                view._start_workflow()
                print("built:", " ".join(sent.get("args", [])))
                missing = [f for f in flags if f not in sent.get("args", [])]
                print("flags not in the Studio command:", missing or "none")
                if args.dry_run or "args" not in sent:
                    win.close()
                    app.quit()
                    return
                st["phase"], st["run_t"] = "run", time.monotonic()
                # Status, step bars and the top of the console.
                bar.setValue(view.status_title.mapTo(scroll.widget(), QPoint(0, 0)).y() - 24)
            QTimer.singleShot(LIVE_MS, tick)
            return
        if st["done_t"] is not None:
            grab("end")
            if time.monotonic() - st["done_t"] > 10.0:
                win.close()
                app.quit()
                return
            QTimer.singleShot(LIVE_MS, tick)
            return
        live = time.monotonic() - st["run_t"] < args.live_sec
        grab("live" if live else "timelapse")
        QTimer.singleShot(LIVE_MS if live else SLOW_MS, tick)

    QTimer.singleShot(1500, tick)
    app.exec()
    if not args.dry_run:
        assemble(frames, out_dir, tuple(int(x) for x in args.video_size.split("x")))


if __name__ == "__main__":
    main()
