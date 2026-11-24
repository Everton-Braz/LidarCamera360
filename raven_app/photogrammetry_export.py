"""Optional hand-off of a 3DGS export to RealityScan or Metashape, for people who own them.

Neither program is bundled or required. Each photo stays one fisheye image (no pinhole faces), resampled once
to the projection the target program models exactly, at the poses, scale and frame of the LiDAR scan:

* Metashape: the equidistant fisheye, its own `Fisheye` sensor type, with every distortion term at zero;
* RealityScan: the stereographic fisheye, which is its `division` lens with a single coefficient.

    <colmap_3dgs>/metashape/    images/ masks/ lidar.ply cameras.json import_metashape.py
    <colmap_3dgs>/realityscan/  images/*.jpg + .xmp (pose and lens) + .jpg.mask.png, lidar.las
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from raven_app import openmvs_dense as od

PROGRAMS = {'realityscan': 'RealityScan', 'metashape': 'Metashape'}
STEREOGRAPHIC_K1 = -0.25       # tan(theta) = r / (1 - r^2/4) when r = 2 tan(theta/2)


def find_program(program, configured=''):
    """Path of the installed program, or None. `configured` (Settings) wins."""
    if configured and Path(configured).is_file():
        return Path(configured)
    base = Path(os.environ.get('ProgramFiles', r'C:\Program Files'))
    patterns = {'realityscan': ('Epic Games/RealityScan*/RealityScan.exe', 'Capturing Reality/RealityCapture/RealityCapture.exe'),
                'metashape': ('Agisoft/Metashape Pro/metashape.exe', 'Agisoft/Metashape/metashape.exe')}[program]
    for pattern in patterns:
        found = sorted(base.glob(pattern), reverse=True)
        if found:
            return found[0]
    return None


def _flat_names(images):
    names = [Path(n).with_suffix('').as_posix().replace('/', '_') for n, *_ in images]
    if len(set(names)) != len(names):
        raise ValueError('two photos of the export share a name')
    return names


def _load(colmap_dir):
    cameras, images = od.read_colmap_text(colmap_dir / 'sparse' / '0')
    images = [i for i in images if (colmap_dir / 'images' / i[0]).is_file()]
    if not images:
        raise ValueError(f'{colmap_dir} has no registered photo')
    return cameras, images, _flat_names(images)


def _fresh(folder):
    """Remove the previous export's photos from `folder`, keeping any project the user saved in it."""
    for sub in ('images', 'masks'):
        if (folder / sub).is_dir():
            shutil.rmtree(folder / sub)
    folder.mkdir(parents=True, exist_ok=True)


METASHAPE_SCRIPT = '''"""Builds a Metashape project from this folder: photos, masks, fixed fisheye lens and poses, LiDAR.

Metashape: Tools > Run Script..., or   metashape.exe -r import_metashape.py
"""
import json, os
import Metashape

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "cameras.json"), encoding="utf-8") as f:
    data = json.load(f)
app_doc = Metashape.app.document
doc = Metashape.Document() if app_doc is None or len(app_doc.chunks) else app_doc
chunk = doc.addChunk()
chunk.label = "LidarCamera360"
chunk.addPhotos([os.path.join(here, "images", c["image"]) for c in data["cameras"]])
sensor = chunk.sensors[0]
sensor.type = Metashape.Sensor.Type.Fisheye
cal = Metashape.Calibration()
cal.type = Metashape.Sensor.Type.Fisheye
cal.width = cal.height = data["size"]
cal.f = data["focal"]
sensor.user_calib = cal
sensor.fixed_calibration = True
poses = {os.path.splitext(c["image"])[0]: c["camera_to_world"] for c in data["cameras"]}
chunk.crs = Metashape.CoordinateSystem('LOCAL_CS["Local Coordinates (m)",LOCAL_DATUM["Local Datum",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]]]')
for cam in chunk.cameras:
    cam.sensor = sensor
    cam.transform = Metashape.Matrix(poses[os.path.splitext(os.path.basename(cam.photo.path))[0]])
chunk.transform.matrix = Metashape.Matrix.Diag([1, 1, 1, 1])
chunk.generateMasks(path=os.path.join(here, "masks", "{filename}.png"), masking_mode=Metashape.MaskingModeFile, cameras=chunk.cameras)
lidar = os.path.join(here, "lidar.ply")
if os.path.isfile(lidar):
    chunk.importPointCloud(path=lidar, format=Metashape.PointCloudFormatPLY)
doc.save(os.path.join(here, "lidarcamera360.psx"))
print("LidarCamera360: %d photos imported at the LiDAR poses" % len(chunk.cameras))
'''


def export_metashape(colmap_dir, size=od.QUALITY['full'], out_dir=None, log=print):
    colmap_dir = Path(colmap_dir)
    cameras, images, names = _load(colmap_dir)
    out = Path(out_dir) if out_dir else colmap_dir / 'metashape'
    _fresh(out)
    od.prepare_images(colmap_dir, cameras, images, size, out, names=names, mask_name='{}.png', log=log)
    entries = []
    for name, (_, _, R, C) in zip(names, images):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R.T, C
        entries.append({'image': f'{name}.jpg', 'camera_to_world': T.tolist()})
    (out / 'cameras.json').write_text(json.dumps({'size': size, 'focal': size / od.FISHEYE_FOV, 'cameras': entries}, indent=1),
                                      encoding='utf-8')
    (out / 'import_metashape.py').write_text(METASHAPE_SCRIPT, encoding='utf-8')
    seed = od.find_lidar_seed(colmap_dir)
    if seed:
        shutil.copyfile(seed, out / 'lidar.ply')
    log(f'[+] Metashape export: {out}')
    return out


XMP = '''<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description xmlns:xcr="http://www.capturingreality.com/ns/xcr/1.1#" xcr:Version="3"
       xcr:PosePrior="{prior}" xcr:Coordinates="absolute" xcr:DistortionModel="division"
       xcr:FocalLength35mm="{focal35:.12g}" xcr:Skew="0" xcr:AspectRatio="1"
       xcr:PrincipalPointU="0" xcr:PrincipalPointV="0" xcr:CalibrationPrior="{lens_prior}"
       xcr:CalibrationGroup="1" xcr:DistortionGroup="1" xcr:InTexturing="1" xcr:InMeshing="1">
      <xcr:Rotation>{rotation}</xcr:Rotation>
      <xcr:Position>{position}</xcr:Position>
      <xcr:DistortionCoeficients>{k1:.12g} 0 0 0 0 0</xcr:DistortionCoeficients>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
'''


def realityscan_xmp(R, C, size, prior='locked', lens_prior='exact', k1=None):
    focal = od.stereographic_focal(size)
    return XMP.format(prior=prior, lens_prior=lens_prior, focal35=36.0 * focal / size,
                      k1=STEREOGRAPHIC_K1 if k1 is None else k1,
                      rotation=' '.join(f'{a:.12g}' for a in np.asarray(R).ravel()),
                      position=' '.join(f'{a:.12g}' for a in np.asarray(C)))


def export_realityscan(colmap_dir, size=3072, out_dir=None, log=print):
    colmap_dir = Path(colmap_dir)
    cameras, images, names = _load(colmap_dir)
    out = Path(out_dir) if out_dir else colmap_dir / 'realityscan'
    _fresh(out)
    od.prepare_images(colmap_dir, cameras, images, size, out, names=names, mask_dir='images',
                      mask_name='{}.jpg.mask.png', rays_of=od.stereographic_rays,
                      focal=od.stereographic_focal(size), log=log)
    for name, (_, _, R, C) in zip(names, images):
        (out / 'images' / f'{name}.xmp').write_text(realityscan_xmp(R, C, size), encoding='utf-8')
    seed = od.find_lidar_seed(colmap_dir)
    if seed:
        try:
            from raven_app.cloud_export import _write_las
            from raven_app.cloud_io import load_cloud
            _write_las(load_cloud(seed), out / 'lidar.las')
        except Exception as exc:
            log(f'[!] The LiDAR cloud was not written for RealityScan: {exc}')
    log(f'[+] RealityScan export: {out}')
    return out


def export_for(program, colmap_dir, log=print, **options):
    return {'realityscan': export_realityscan, 'metashape': export_metashape}[program](colmap_dir, log=log, **options)


def open_commands(program, executable, folder):
    """Commands to run in order; the last one opens the program's own window.

    Metashape runs a script without a window, so the project is built first. RealityScan would otherwise
    replace the division lens by its default Brown fit, which cannot follow a fisheye."""
    if program == 'metashape':
        return [[str(executable), '-r', str(folder / 'import_metashape.py')],
                [str(executable), str(folder / 'lidarcamera360.psx')]]
    return [[str(executable), '-set', 'sfmDistortionModel=Division', '-addFolder', str(folder / 'images')]]


def open_in(program, folder, configured='', log=print):
    """Open an exported folder. Blocks while Metashape imports, so call it from a worker."""
    folder = Path(folder)
    executable = find_program(program, configured)
    if executable is None:
        raise FileNotFoundError(f'{PROGRAMS[program]} is not installed; the files are in {folder}')
    *before, last = open_commands(program, executable, folder)
    for command in before:
        done = subprocess.run(command, cwd=folder, capture_output=True, text=True, errors='replace')
        if done.returncode or not (folder / 'lidarcamera360.psx').is_file():
            raise RuntimeError(f'{PROGRAMS[program]} could not import {folder}: {(done.stdout + done.stderr)[-600:]}')
    log(f'[*] Opening {PROGRAMS[program]}')
    return subprocess.Popen(last, cwd=folder)
