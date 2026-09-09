"""Reproducible synthetic Vulkan throughput check, including bridge IO and JPEG decoding."""
import argparse, json, time
from pathlib import Path
import cv2
import numpy as np
from raven_app.vulkan_engine import colorize_views

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--points',type=int,default=10000000)
    p.add_argument('--frames',type=int,default=175)
    p.add_argument('--output',type=Path,default=Path('build/vulkan-benchmark'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(23)
    points=rng.uniform(-10,10,(a.points,3));points[:,2]+=12
    path=a.output/'frame.jpg'
    cv2.imwrite(str(path),np.full((3840,3840,3),(30,80,150),np.uint8))
    params=[1080.19,1080.19,1920,1920]+[0]*8
    views=[(path,np.eye(3),np.array([i*.03,0,0]),params) for i in range(a.frames)]
    start=time.perf_counter();rgb=colorize_views(points,views,a.output);elapsed=time.perf_counter()-start
    report={'kind':'synthetic, not the real scan acceptance benchmark','points':a.points,'frames':a.frames,'image_size':[3840,3840],'seconds_including_bridge_io':elapsed,'success':rgb is not None,'under_3_seconds':elapsed<3}
    (a.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if rgb is None:raise SystemExit(1)

if __name__=='__main__':main()
