"""Lazy imports keep --help/--version fast and headless commands GUI-free."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from raven_app import __version__


def resources():
    return Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[1]))


def engine():
    packaged=resources()/'bin/fastlivo2.exe'
    return packaged if packaged.is_file() else resources()/'build/native/Release/fastlivo2.exe'


def positive(value):
    import math
    v=float(value)
    if not math.isfinite(v) or v<=0: raise argparse.ArgumentTypeError('must be finite and positive')
    return v


def finite(value):
    import math
    v=float(value)
    if not math.isfinite(v):raise argparse.ArgumentTypeError('must be finite')
    return v


def bag_options(p):
    p.add_argument('--bag',type=Path,nargs='+',required=True,help='One merged bag, or non-overlapping split bags')
    p.add_argument('--lidar-topic',default='/vanjee_722z')
    p.add_argument('--imu-topic',default='/vanjee_imu_packets')
    p.add_argument('--image-topic',default='/camera_front/image_raw')
    p.add_argument('--lio',action='store_true',help='LiDAR + IMU, without camera measurements')
    p.add_argument('--time-field',default='timestamp')
    p.add_argument('--time-unit',choices=['seconds','milliseconds','microseconds','nanoseconds'],default='seconds')
    p.add_argument('--time-origin',choices=['first-point','header','relative'],default='first-point')


def parse(argv=None):
    p=argparse.ArgumentParser(prog='RavenCalibrator',description='Standalone Raven LiDAR / camera processor')
    p.add_argument('--version',action='version',version='RavenCalibrator '+__version__)
    p.add_argument('--headless',action='store_true',help='Require a CLI command; never open the desktop UI')
    sub=p.add_subparsers(dest='command')
    sub.add_parser('gui',help='Open desktop controls')
    sub.add_parser('doctor',help='Report bundled engines and numerical runtime')
    info=sub.add_parser('inspect-bag',help='List bag topics without ROS')
    info.add_argument('--bag',type=Path,nargs='+',required=True)
    export=sub.add_parser('export-bag',help='Convert bags to native FLV2 input')
    bag_options(export);export.add_argument('--output',type=Path,required=True)
    slam=sub.add_parser('slam',help='Run native FAST-LIVO2 directly from bags')
    bag_options(slam)
    slam.add_argument('--config',type=Path,default=resources()/'FAST-LIVO2/config/raven.yaml')
    slam.add_argument('--camera',type=Path,default=resources()/'FAST-LIVO2/config/camera_raven.yaml')
    slam.add_argument('--output',type=Path,required=True)
    slam.add_argument('--threads',type=int,choices=range(1,257),metavar='1..256',default=min(4,os.cpu_count() or 1))
    color=sub.add_parser('colorize',help='Run the existing calibration and colorization pipeline')
    color.add_argument('--dataset',type=Path,required=True)
    color.add_argument('--method',choices=['sfm','direct','all'],default='direct')
    color.add_argument('--fps',type=positive,default=1.)
    color.add_argument('--calib','--calib-json',type=Path)
    color.add_argument('--dt',type=finite)
    color.add_argument('--run-spirula',action='store_true')
    color.add_argument('--recalibrate-from-sfm',action='store_true')
    wf=sub.add_parser('workflow',help='Run end-to-end processing: Bag + INSV -> SLAM -> Sync -> Colorize -> Deliverables')
    wf.add_argument('--bag',type=Path,required=True,help='LiDAR ROS bag file')
    wf.add_argument('--insv',type=Path,required=True,help='Insta360 video file')
    wf.add_argument('--output',type=Path,required=True,help='Output directory for deliverables')
    wf.add_argument('--lidar-topic',default='/vanjee_722z')
    wf.add_argument('--imu-topic',default='/vanjee_imu_packets')
    wf.add_argument('--lio',action='store_true',default=True,help='LiDAR + IMU SLAM')
    wf.add_argument('--threads',type=int,default=min(4,os.cpu_count() or 1))
    wf.add_argument('--fps',type=positive,default=1.)
    wf.add_argument('--method',choices=['direct','sfm','all'],default='direct')
    wf.add_argument('--calib','--calib-json',type=Path)
    wf.add_argument('--dt',type=finite)
    wf.add_argument('--export-ply',action='store_true',default=False)
    wf.add_argument('--export-pcd',action='store_true',default=False)
    wf.add_argument('--export-colmap',action='store_true',default=False)
    a=p.parse_args(argv)
    if a.headless and a.command in (None,'gui'):p.error('--headless requires a processing or inspection command')
    return a


def export_options(a):
    return {k:getattr(a,k) for k in ('lidar_topic','imu_topic','image_topic','lio','time_field','time_unit','time_origin')}


def run(a):
    if a.command in (None,'gui'):
        from raven_app.gui import launch
        launch();return 0
    if a.command=='doctor':
        import numpy, scipy, cv2
        report={'version':__version__,'native_engine':str(engine()),'native_engine_exists':engine().is_file(),
                'numpy':numpy.__version__,'scipy':scipy.__version__,'opencv':cv2.__version__,
                'spirula_exists':(resources()/'spirula/spirula.exe').is_file()}
        if engine().is_file():
            probe=subprocess.run([str(engine()),'--version'],capture_output=True,text=True)
            report['native_exit_code']=probe.returncode
            report['native_version']=probe.stdout.strip()
        print(json.dumps(report,indent=2));return 0 if report.get('native_exit_code')==0 else 2
    if a.command=='inspect-bag':
        from raven_app.bag_io import inspect_bags
        print(json.dumps(inspect_bags(a.bag),indent=2));return 0
    if a.command=='export-bag':
        from raven_app.bag_io import export_bags
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('xb') as f: result=export_bags(a.bag,f,**export_options(a))
        print(json.dumps(result));return 0
    if a.command=='slam':
        from raven_app.bag_io import export_bags
        if not engine().is_file():raise FileNotFoundError('Native engine missing; run tools/build_windows.ps1')
        cmd=[str(engine()),'--input','-','--config',str(a.config),'--camera',str(a.camera),
             '--output',str(a.output),'--threads',str(a.threads)]
        if a.lio:cmd.append('--lio')
        with subprocess.Popen(cmd,stdin=subprocess.PIPE) as child:
            try:
                export_bags(a.bag,child.stdin,**export_options(a))
                child.stdin.close()
                return child.wait()
            except BaseException:
                try:child.stdin.close()
                except OSError:pass
                # EOF permits a normal flush; malformed input must not report success.
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.terminate();child.wait()
                raise
    if a.command=='colorize':
        from scripts import pipeline_auto_calibrator_and_colorizer as pipeline
        dataset=a.dataset.resolve()
        for f in ('slam_out/pcd/all_raw_points.pcd','slam_out/result/Raven_3DMakerPro_Scan.txt'):
            if not (dataset/f).is_file():raise FileNotFoundError(f'Missing dataset input: {f}')
        for cam in ('cam0','cam1'):
            if not any((dataset/'images'/cam).glob('*.jpg')):raise ValueError(f'Missing extracted frames in images/{cam}')
        if a.run_spirula and not pipeline.run_spirula_sfm_auto(dataset):raise RuntimeError('Spirula reconstruction failed')
        if a.recalibrate_from_sfm:pipeline.recalibrate_from_sfm(dataset,fps=a.fps)
        if a.method in ('sfm','all'):pipeline.colorize_via_spirula_sfm(dataset,fps=a.fps)
        if a.method in ('direct','all'):pipeline.colorize_via_direct_rigid(dataset,a.calib,fps=a.fps,dt_override=a.dt)
        return 0
    if a.command=='workflow':
        from raven_app.workflow import execute_unified_workflow
        return execute_unified_workflow(
            a.bag, a.insv, a.output,
            lio=a.lio,
            threads=a.threads,
            lidar_topic=a.lidar_topic,
            imu_topic=a.imu_topic,
            fps=a.fps,
            method=a.method,
            calib_json=a.calib,
            dt_override=a.dt,
            export_ply=a.export_ply,
            export_pcd=a.export_pcd,
            export_colmap=a.export_colmap
        )
    raise ValueError('Unknown command')


def main(argv=None):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='replace',line_buffering=True)
    if hasattr(signal,'SIGBREAK'):
        signal.signal(signal.SIGBREAK,signal.default_int_handler)
    try:return run(parse(argv))
    except KeyboardInterrupt:
        print('Cancelled.',file=sys.stderr);return 130
    except Exception as e:
        print(f'RavenCalibrator: {e}',file=sys.stderr);return 2
