"""Windows integration: cancel the app's process group and verify native flush."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--bag',required=True,type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[1]
output=root/'build/validation'/('cancel-'+str(time.time_ns()))
log=root/'build/cancel-check.log'
with log.open('w',encoding='utf-8') as f:
    child=subprocess.Popen([str(root/'dist/RavenCalibrator/RavenCalibrator.exe'),'--headless','slam',
        '--bag',str(a.bag),'--lio','--output',str(output)],stdout=f,stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        deadline=time.monotonic()+20
        while time.monotonic()<deadline and child.poll() is None:
            if 'Update Voxel Map' in log.read_text(encoding='utf-8',errors='replace'):break
            time.sleep(.1)
        assert child.poll() is None,'Job finished before cancellation was sent'
        child.send_signal(signal.CTRL_BREAK_EVENT)
        code=child.wait(timeout=35)
        assert code==130,(code,log.read_text(encoding='utf-8',errors='replace')[-2000:])
        report=json.loads((output/'run.json').read_text())
        assert report['cancelled'] and report['points']>0,report
        assert (output/'pcd/all_raw_points.pcd').is_file()
        print(json.dumps({'exit_code':code,'partial_map_saved':True,**report},indent=2))
    finally:
        if child.poll() is None:child.terminate();child.wait()
