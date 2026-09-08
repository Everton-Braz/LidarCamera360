"""Exercise native parser failures and configuration validation."""
from pathlib import Path
import struct
import subprocess
import tempfile


ROOT=Path(__file__).resolve().parents[1]
EXE=ROOT/'build/native/Release/fastlivo2.exe'
with tempfile.TemporaryDirectory(prefix='raven-native-') as tmp:
    tmp=Path(tmp)
    cases=[b'not-flv2',b'FLV2\x01\0\0\0'+struct.pack('<BdI',1,1.,6),
           b'FLV2\x01\0\0\0'+struct.pack('<BdI',2,1.,10000001),
           b'FLV2\x01\0\0\0'+struct.pack('<BdI',3,float('nan'),0)]
    for i,data in enumerate(cases):
        source=tmp/f'bad-{i}.flv2';source.write_bytes(data)
        command=[str(EXE),'--input',str(source),'--config',str(ROOT/'FAST-LIVO2/config/raven.yaml'),
                 '--camera',str(ROOT/'FAST-LIVO2/config/camera_raven.yaml'),'--output',str(tmp/f'out-{i}'),'--lio']
        result=subprocess.run(command,capture_output=True,text=True,timeout=20)
        assert result.returncode==2,(i,result.returncode,result.stdout,result.stderr)
        assert not (tmp/f'out-{i}/pcd/all_raw_points.pcd').exists()
    print(f'Native malformed-input checks passed: {len(cases)}')
