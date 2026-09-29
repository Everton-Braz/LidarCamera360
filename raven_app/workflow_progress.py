"""Parsing and lightweight ETA estimates for the unified workflow UI."""
import json
import re
import statistics
from pathlib import Path


STAGE_NAMES = (
    'Extract video frames',
    'Synchronize timestamps',
    'FAST-LIVO2 SLAM',
    'SfM alignment and rig calibration',
    'Generate person masks',
    'Point-cloud colorization',
    'Georeferencing and 3DGS deliverables',
)

_STAGE_LINE = re.compile(r'^\[STAGE\s+(\d+)/(\d+)\]\s*(.*)$')
_VULKAN_LINE = re.compile(r'Vulkan frames\s+(\d+)/(\d+)', re.IGNORECASE)
_PROJECTED_LINE = re.compile(r'\[(\d+)\s*/\s*(\d+)\]\s+Projected frames', re.IGNORECASE)
_MASKS_READY = re.compile(
    r'(?:\[\+\]\s+\d+ combined foreground masks ready|'
    r'\[\*\]\s+Reusing \d+ combined foreground masks)', re.IGNORECASE
)


def parse_progress_line(line):
    """Return a compact event parsed from a workflow/native progress line."""
    text = str(line).strip()
    match = _STAGE_LINE.match(text)
    if match:
        stage, total = int(match.group(1)), int(match.group(2))
        return {'kind': 'stage', 'stage': stage, 'total': total,
                'name': match.group(3).rstrip('. ')}
    match = _VULKAN_LINE.search(text) or _PROJECTED_LINE.search(text)
    if match:
        done, total = int(match.group(1)), int(match.group(2))
        if total > 0:
            return {'kind': 'stage_fraction', 'fraction': max(0., min(1., done / total))}
    if _MASKS_READY.search(text):
        return {'kind': 'masks_ready'}
    if '[SUCCESS] Unified Workflow Complete' in text:
        return {'kind': 'complete'}
    return None


def _pcd_point_count(path):
    try:
        with Path(path).open('rb') as stream:
            for _ in range(64):
                line = stream.readline()
                if not line:
                    break
                fields = line.decode('ascii', errors='ignore').strip().split()
                if len(fields) == 2 and fields[0].upper() in {'POINTS', 'WIDTH'}:
                    return max(0, int(fields[1]))
                if fields and fields[0].upper() == 'DATA':
                    break
    except (OSError, ValueError):
        pass
    return 0


def estimate_stage_seconds(output_dir, bag_path, insv_path, *,
                            use_masks=True, use_vulkan=True, export_colmap=True,
                            method='sfm', recalibrate=True):
    """Estimate stage durations from cached artifacts, input sizes, and workload.

    The mask and Vulkan rates use the latest local RF-DETR/Vulkan workflow run;
    the UI labels resulting times as approximate and learns per-stage times.
    """
    root = Path(output_dir)
    bag = Path(bag_path) if bag_path else None
    video = Path(insv_path) if insv_path else None
    images_dir = root / 'images'
    image_count = 0
    for camera in ('cam0', 'cam1'):
        try:
            image_count += sum(1 for p in (images_dir / camera).iterdir()
                               if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        except OSError:
            pass
    if image_count < 2:
        video_bytes = video.stat().st_size if video and video.is_file() else 0
        # A conservative fallback for a first run before frame extraction.
        image_count = max(2, round(video_bytes / (593 * 1024 * 1024) * 3600)) if video_bytes else 3600

    raw_pcd = root / 'slam_out' / 'pcd' / 'all_raw_points.pcd'
    trajectory = root / 'slam_out' / 'result' / 'Raven_3DMakerPro_Scan.txt'
    cached_slam = raw_pcd.is_file() and trajectory.is_file()
    run_json = root / 'slam_out' / 'run.json'
    points = _pcd_point_count(raw_pcd)
    if not points and run_json.is_file():
        try:
            points = int(json.loads(run_json.read_text(encoding='utf-8')).get('points', 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            points = 0
    if not points and bag and bag.is_file():
        points = int(bag.stat().st_size * 0.031)
    points = max(points, 1_000_000)

    frames_ready = any((images_dir / camera).is_dir() for camera in ('cam0', 'cam1')) and image_count > 2
    sparse_ready = (root / 'sparse' / '0' / 'points3D.bin').is_file()
    sparse_work = 642. * image_count / 3579. if not sparse_ready and method in {'sfm', 'all'} else 0.
    bag_bytes = bag.stat().st_size if bag and bag.is_file() else 1_000_000_000
    video_bytes = video.stat().st_size if video and video.is_file() else 600_000_000
    if sparse_ready:
        alignment_seconds = 30.0 if recalibrate or method in {'sfm', 'all'} else 5.0
    elif method in {'sfm', 'all'}:
        alignment_seconds = max(120.0, sparse_work + 30.0)
    else:
        alignment_seconds = 5.0

    return [
        1.0 if frames_ready else max(30., video_bytes / (5 * 1024 * 1024)),
        20.0,
        1.0 if cached_slam else max(10., bag_bytes / (16 * 1024 * 1024)),
        alignment_seconds,
        image_count / 9.0 if use_masks else 0.0,
        image_count / 7.24 * (points / 30_301_126) * (1.0 if use_vulkan else 3.0),
        (60.0 + points / 100_000.) if export_colmap else 30.0,
    ]


def load_stage_history(output_dir):
    path = Path(output_dir) / 'logs' / 'workflow_stage_times.json'
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        stages = data.get('stages', {})
        if '7' not in stages and '6' in stages:
            # Older history combined masks and colorization in stage 5, and
            # stored deliverables in stage 6. Do not reuse the mixed duration.
            return ([list(map(float, stages.get(str(i), [])))[-5:] for i in range(1, 5)]
                    + [[], [], list(map(float, stages.get('6', [])))[-5:]])
        return [list(map(float, stages.get(str(i), [])))[-5:] for i in range(1, 8)]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return [[] for _ in range(7)]


def save_stage_history(output_dir, history):
    path = Path(output_dir) / 'logs' / 'workflow_stage_times.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'stages': {str(i + 1): list(map(float, durations[-5:]))
                          for i, durations in enumerate(history)}}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


def stage_duration_estimates(defaults, history):
    """Prefer median timings from this dataset; fall back to workload estimates."""
    return [float(defaults[i]) if float(defaults[i]) <= 0 else
            (statistics.median(samples) if samples else float(defaults[i]))
            for i, samples in enumerate(history)]
