"""Integration checks against the delivered executable, from an unrelated CWD."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time
import sys

import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);a=p.parse_args()
    exe=a.exe.resolve();root=Path(__file__).resolve().parents[1]
    env=os.environ.copy();env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    env['PATH']=str(Path(os.environ['SystemRoot'])/'System32')
    with tempfile.TemporaryDirectory(prefix='raven-smoke-') as tmp:
        tmp=Path(tmp)
        def call(args,expected=0):
            r=subprocess.run([str(exe),*args],cwd=tmp,env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
            if r.returncode!=expected:raise AssertionError((args,r.returncode,r.stdout,r.stderr))
            return r
        t=time.perf_counter();call(['--help']);startup=time.perf_counter()-t
        gui = call(['gui', '--smoke-test'])
        assert '"gui_ready": true' in gui.stdout, gui.stdout
        report=json.loads(call(['--headless','doctor']).stdout)
        assert report['native_exit_code']==0
        assert report['pyav']
        assert report['spirula_ready']
        assert report['vulkan_colorizer']['ready'], report['vulkan_colorizer']
        sys.path.insert(0, str(root))
        from test_video_vulkan import VideoTests
        source=tmp/'synthetic.insv'
        VideoTests().make_video(source)
        call(['--headless','extract-insv','--insv',str(source),'--output',str(tmp/'extracted')])
        assert len(list((tmp/'extracted/images/cam1').glob('*.jpg')))==3
        call(['--headless'],2)
        call(['colorize','--dataset',str(tmp),'--fps','nan'],2)
        dataset=tmp/'dataset';pcd=dataset/'slam_out/pcd';trj=dataset/'slam_out/result'
        pcd.mkdir(parents=True);trj.mkdir(parents=True)
        calibration=json.loads((root/'calibracao_rigida_raven_insta360.json').read_text())
        transform=np.array(calibration['T_lidar_to_cam0_rigid_4x4'])
        xyz=np.array([[0,0,4],[.2,.1,4],[-.2,-.1,4]],dtype=np.float64)@transform[:3,:3].T+transform[:3,3]
        payload=np.column_stack([xyz,np.zeros(3)]).astype('<f4').tobytes()
        (pcd/'all_raw_points.pcd').write_bytes(b'FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nPOINTS 3\nDATA binary\n'+payload)
        (trj/'Raven_3DMakerPro_Scan.txt').write_text('1000 0 0 0 0 0 0 1\n1001 0 0 0 0 0 0 1\n')
        for cam in ('cam0','cam1'):
            directory=dataset/'images'/cam;directory.mkdir(parents=True)
            image=np.full((3840,3840,3),(20,80,200),dtype=np.uint8)
            assert cv2.imwrite(str(directory/'frame_000001.jpg'),image)
        result=call(['--headless','colorize','--dataset',str(dataset),'--method','direct','--fps','1','--dt','0'])
        assert 'Vulkan frames' in result.stdout, result.stdout
        pcd_files=list((dataset/'deliverables').glob('*.pcd'))
        assert pcd_files, f"No PCD deliverable found in {list((dataset/'deliverables').iterdir())}"
        output=pcd_files[0]
        data=output.read_bytes().split(b'DATA binary\n',1)[1]
        assert len(data)==3*16
        rows=np.frombuffer(data,dtype=[('xyz','<f4',3),('rgb','<u4')])
        assert np.all(rows['rgb']!=0xB4B4B4),'No points received color'
        print(json.dumps({'help_startup_seconds':startup,'doctor':'passed','gui':'passed','invalid_input':'passed',
                          'direct_colorization':'passed','colored_points':3,'isolated_cwd_and_path':True},indent=2))


if __name__=='__main__':main()
