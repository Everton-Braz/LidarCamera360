"""Parsing and lightweight ETA estimates for the unified workflow UI."""
import json
import os
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
_PROJECTED_LINE = re.compile(r'\[(\d+)\s*/\s*(\d+)\]\s+(?:Projected frames|SLAM scans mapped)', re.IGNORECASE)
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


def _bag_seconds(bag):
    """Recording length from the bag index (no message is read), or None."""
    try:
        from rosbags.highlevel import AnyReader
        with AnyReader([Path(bag)]) as reader:
            seconds = reader.duration / 1e9
        return seconds if seconds > 0 else None
    except Exception:
        return None


def _active_slam_dir(root):
    manifest = root / 'active_slam.json'
    try:
        selected = root / json.loads(manifest.read_text(encoding='utf-8'))['relative_directory']
        if selected.is_dir():
            return selected
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return root / 'slam_out'


def estimate_stage_seconds(output_dir, bag_path, insv_path, *,
                            use_masks=True, use_vulkan=True, export_colmap=True,
                            method='sfm', recalibrate=True, fps=None, lenses=2,
                            remove_dynamic=False, rig_refine='none', mask_backend='vulkan'):
    """Estimate stage durations from cached artifacts, input sizes, and workload.

    Rates come from local runs (CUDA decode, Vulkan, TensorRT masks); the UI scales them by what
    this machine measured on earlier runs (machine rates) and prefers this dataset's own timings.
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
    frames_ready = image_count > 2
    video_bytes = video.stat().st_size if video and video.is_file() else 0
    if not frames_ready:
        # Before extraction: one sharp frame per lens per 1/fps seconds of recording.
        seconds = (_bag_seconds(bag) if bag and bag.is_file() else None) or (video_bytes / 7.5e6 if video_bytes else 0)
        image_count = max(2, round(seconds * float(fps or 1.0) * max(1, lenses))) if seconds else 600

    slam_dir = _active_slam_dir(root)
    raw_pcd = slam_dir / 'pcd' / 'all_raw_points.pcd'
    trajectories = list((slam_dir / 'result').glob('*.txt')) if (slam_dir / 'result').is_dir() else []
    cached_slam = raw_pcd.is_file() and bool(trajectories)
    if cached_slam and bag and bag.is_file():
        try:  # a cache made from another recording is not reused
            inputs = json.loads((slam_dir / 'input_config.json').read_text(encoding='utf-8'))
            cached_slam = int(inputs.get('size', -1)) == bag.stat().st_size
        except (OSError, ValueError, TypeError):
            pass
    points = _pcd_point_count(raw_pcd) if cached_slam else 0
    run_json = slam_dir / 'run.json'
    if not points and cached_slam and run_json.is_file():
        try:
            points = int(json.loads(run_json.read_text(encoding='utf-8')).get('points', 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            points = 0
    if not points and bag and bag.is_file():
        points = int(bag.stat().st_size * 0.03)
    points = max(points, 1_000_000)

    sfm = method in {'sfm', 'all'}
    sparse_ready = (root / 'sparse' / '0' / 'points3D.bin').is_file()
    bag_bytes = bag.stat().st_size if bag and bag.is_file() else 300_000_000
    mask_count = 0
    for camera in ('cam0', 'cam1'):
        try:
            mask_count += sum(1 for p in (root / 'masks' / camera).iterdir() if p.suffix.lower() == '.png')
        except OSError:
            pass
    try:
        inference = json.loads((root / 'masks' / 'manifest.json').read_text(encoding='utf-8')).get('inference')
    except (OSError, ValueError, AttributeError):
        inference = None
    masks_reused = frames_ready and mask_count >= image_count and inference is not None
    # Upright tiles (mask_tiles): 22 views per image at ~80/s on TensorRT, 7 at ~8/s on Vulkan.
    mask_seconds = 0.0 if masks_reused or not use_masks else image_count * (0.3 if mask_backend == 'tensorrt' else 1.0)
    if sparse_ready:
        alignment_seconds = 30.0 if recalibrate or sfm else 5.0
    elif sfm:
        alignment_seconds = 15.0 + 0.1 * image_count + mask_seconds  # Spirula SfM needs the person masks first
    else:
        alignment_seconds = 5.0
    masks_in_stage5 = not (sfm and not sparse_ready)
    mask_stage = (mask_seconds if masks_in_stage5 else 0.0) + (points / 55_000. if remove_dynamic else 0.0)
    colorize = (20.0 + points / 150_000. + image_count / 7.24 * (points / 30_301_126) * (1.0 if use_vulkan else 3.0)
                + (80.0 if rig_refine and rig_refine != 'none' else 0.0)
                # standing-person vote: ~0.3 s per view of a 5.5 M point cloud on 6 threads
                + (image_count * points / 18.5e6 if remove_dynamic and use_masks else 0.0))

    return [
        1.0 if frames_ready else (max(20., video_bytes / 22e6) if video_bytes else 0.1 * image_count),
        5.0,
        1.0 if cached_slam else max(10., bag_bytes / 2.8e6),
        alignment_seconds,
        mask_stage,
        colorize,
        (20.0 + points / 200_000. + 0.05 * image_count) if export_colmap else 20.0,
    ]


MAX_STAGE_RATIOS = 8
STAGE_RATIO_LIMITS = (0.2, 5.0)
MIN_LEARNED_STAGE_SECONDS = 5.0


def dataset_times_path(output_dir):
    return Path(output_dir) / 'logs' / 'workflow_stage_times.json'


def machine_rates_path():
    """Measured/estimated stage ratios of this computer, shared by every dataset."""
    base = os.environ.get('APPDATA')
    return (Path(base) if base else Path.home() / '.config') / 'LidarCamera360' / 'workflow_stage_rates.json'


def _read_json(path):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_stage_ratios(path):
    """Per-stage lists of actual/estimated duration ratios saved at ``path``."""
    stored = _read_json(path).get('ratios', {})
    low, high = STAGE_RATIO_LIMITS
    ratios = []
    for i in range(1, len(STAGE_NAMES) + 1):
        try:
            values = [min(high, max(low, float(v))) for v in stored.get(str(i), [])]
        except (TypeError, ValueError, AttributeError):
            values = []
        ratios.append(values[-MAX_STAGE_RATIOS:])
    return ratios


def save_stage_ratios(path, ratios):
    data = _read_json(path)
    data['ratios'] = {str(i + 1): [round(float(v), 4) for v in values[-MAX_STAGE_RATIOS:]]
                      for i, values in enumerate(ratios)}
    _write_json(path, data)


def add_stage_ratio(ratios, index, actual_seconds, estimated_seconds):
    """Learn how this run compared with the workload estimate. Trivial or skipped stages teach nothing."""
    if estimated_seconds <= MIN_LEARNED_STAGE_SECONDS or actual_seconds < 1.0:
        return False  # a step that ends within a second reused its outputs
    low, high = STAGE_RATIO_LIMITS
    ratio = min(high, max(low, actual_seconds / estimated_seconds))
    ratios[index] = (list(ratios[index]) + [ratio])[-MAX_STAGE_RATIOS:]
    return True


def load_stage_history(output_dir):
    path = dataset_times_path(output_dir)
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
    path = dataset_times_path(output_dir)
    data = _read_json(path)
    data['stages'] = {str(i + 1): list(map(float, durations[-5:])) for i, durations in enumerate(history)}
    _write_json(path, data)


def stage_duration_estimates(defaults, history, dataset_ratios=None, machine_ratios=None):
    """Scale each workload estimate by what was measured before.

    Order: this dataset's ratios, this dataset's older absolute timings, this computer's ratios
    from other datasets, then the plain estimate. Cached, skipped and trivial stages keep their estimate.
    """
    estimates = []
    for i, default in enumerate(map(float, defaults)):
        own = dataset_ratios[i] if dataset_ratios else []
        legacy = history[i] if history and i < len(history) else []
        machine = machine_ratios[i] if machine_ratios else []
        if default <= MIN_LEARNED_STAGE_SECONDS:
            estimates.append(default)
        elif own:
            estimates.append(default * statistics.median(own))
        elif legacy:
            estimates.append(statistics.median(legacy))
        elif machine:
            estimates.append(default * statistics.median(machine))
        else:
            estimates.append(default)
    return estimates
