"""Lazy, hash-verified RF-DETR model and TensorRT runtime resources."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

from raven_app.config import get_app_root, get_config_dir


MODEL_NAME = 'rfdetr-seg-medium.onnx'
MODEL_SIZE = 139_278_517
MODEL_SHA256 = 'ac764d38d19183940d120f3333eb769dbca90100b6955f2cef2b3848565f7cf4'
MODEL_URL = 'https://huggingface.co/davidkodar/rf-detr-seg-medium-onnx/resolve/main/rf_detr_seg_medium_v1.onnx'

TENSORRT_VERSION = '10.13.3.9'
TENSORRT_WHEEL_NAME = 'tensorrt_cu13_libs-10.13.3.9-py2.py3-none-win_amd64.whl'
TENSORRT_WHEEL_SIZE = 1_327_112_482
TENSORRT_WHEEL_SHA256 = '2561ed2f0a79c011701c9e74e0c5ccac5f84c19f5c51d17986b55d9adb4f2d21'
TENSORRT_WHEEL_URL = f'https://pypi.nvidia.com/tensorrt-cu13-libs/{TENSORRT_WHEEL_NAME}'
TENSORRT_DLLS = {
    'nvinfer_10.dll': 369_255_968,
    'nvonnxparser_10.dll': 3_045_408,
    'nvinfer_builder_resource_10.dll': 1_139_393_056,
    'nvinfer_plugin_10.dll': 46_949_408,
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _valid_file(path, size, digest):
    path = Path(path)
    return path.is_file() and path.stat().st_size == size and _sha256(path) == digest


def resolve_model(model_path=None):
    """Find an explicit, user-cached, or development model without downloading."""
    if model_path is not None:
        candidate = Path(model_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f'RF-DETR model not found: {candidate}')
        return candidate.resolve()
    configured = os.environ.get('RAVEN_RFDETR_MODEL', '').strip()
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f'RAVEN_RFDETR_MODEL does not exist: {candidate}')
        return candidate.resolve()
    root = get_app_root()
    candidates = (get_config_dir() / 'models' / MODEL_NAME, root / 'models' / MODEL_NAME)
    return next((path.resolve() for path in candidates if path.is_file()), None)


def resolve_masker_executable():
    root = get_app_root()
    candidates = (root / 'bin/rfdetr-masker.exe',
                  root / 'build/rfdetr-masker/Release/rfdetr-masker.exe')
    return next((path.resolve() for path in candidates if path.is_file()), None)


def resolve_runtime_dir():
    """Find a complete TensorRT runtime, honoring an optional user override."""
    root = get_app_root()
    candidates = []
    override = os.environ.get('RAVEN_RFDETR_RUNTIME', '').strip()
    if override:
        path = Path(override).expanduser()
        candidates.extend((path, path / 'bin'))
    candidates.extend((
        root / 'bin',
        root / 'build/rfdetr-masker/Release',
        get_config_dir() / f'rfdetr-runtime-{TENSORRT_VERSION}' / 'bin',
    ))
    for directory in candidates:
        if all((directory / name).is_file() for name in TENSORRT_DLLS):
            return directory.resolve()
    return None


def masker_environment(executable=None, base=None):
    """Return an environment that can load TensorRT from its cache directory."""
    env = dict(os.environ if base is None else base)
    directories = []
    runtime = resolve_runtime_dir()
    if runtime is not None:
        directories.append(str(runtime))
    if executable is not None:
        directories.append(str(Path(executable).resolve().parent))
    old_path = env.get('PATH', '')
    env['PATH'] = os.pathsep.join(dict.fromkeys(directories + ([old_path] if old_path else [])))
    return env


def _download(url, target, expected_size, expected_sha256, resource, progress=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if _valid_file(target, expected_size, expected_sha256):
        if progress:
            progress(resource, 100)
        return target
    lock = target.with_name(target.name + '.lock')
    deadline = time.monotonic() + 6 * 60 * 60
    descriptor = None
    while descriptor is None:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if not lock.exists() or (time.time() - lock.stat().st_mtime) > 6 * 60 * 60:
                lock.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Another RF-DETR resource download is still active: {resource}')
            time.sleep(1)
            if not lock.exists() and _valid_file(target, expected_size, expected_sha256):
                if progress:
                    progress(resource, 100)
                return target
    os.write(descriptor, str(os.getpid()).encode('ascii'))
    os.close(descriptor)
    temporary = target.with_name(target.name + '.part')
    try:
        if _valid_file(target, expected_size, expected_sha256):
            if progress:
                progress(resource, 100)
            return target
        temporary.unlink(missing_ok=True)
        digest = hashlib.sha256()
        received = 0
        last_percent = -1
        request = urllib.request.Request(url, headers={'User-Agent': 'LidarCamera360/0.2 RF-DETR resource setup'})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as output:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
                received += len(block)
                percent = min(100, received * 100 // expected_size)
                if progress and (percent != last_percent):
                    progress(resource, percent)
                    last_percent = percent
        if received != expected_size:
            raise OSError(f'{resource} download size mismatch: expected {expected_size:,} bytes, received {received:,}')
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha256:
            raise OSError(f'{resource} SHA-256 mismatch: {actual_sha}')
        temporary.replace(target)
        if progress and last_percent < 100:
            progress(resource, 100)
        return target
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        lock.unlink(missing_ok=True)


def ensure_model(progress=None):
    target = get_config_dir() / 'models' / MODEL_NAME
    configured = os.environ.get('RAVEN_RFDETR_MODEL', '').strip()
    if configured:
        return resolve_model(configured)
    if _valid_file(target, MODEL_SIZE, MODEL_SHA256):
        return target.resolve()
    bundled = get_app_root() / 'models' / MODEL_NAME
    if _valid_file(bundled, MODEL_SIZE, MODEL_SHA256):
        return bundled.resolve()
    return _download(MODEL_URL, target, MODEL_SIZE, MODEL_SHA256, 'model', progress).resolve()


def _extract_runtime(wheel, target_dir, progress=None):
    target_dir = Path(target_dir)
    staging = target_dir.parent / (target_dir.name + '.staging')
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        extracted = 0
        total_runtime = sum(TENSORRT_DLLS.values())
        with zipfile.ZipFile(wheel) as archive:
            entries = {}
            for info in archive.infolist():
                name = Path(info.filename).name
                if name in TENSORRT_DLLS and info.filename == f'tensorrt_libs/{name}':
                    entries[name] = info
            if set(entries) != set(TENSORRT_DLLS):
                raise OSError('TensorRT package is missing one or more required runtime DLLs')
            for name, expected_size in TENSORRT_DLLS.items():
                info = entries[name]
                if info.file_size != expected_size:
                    raise OSError(f'TensorRT runtime size mismatch for {name}')
                written = 0
                with archive.open(info) as source, (staging / name).open('wb') as output:
                    while True:
                        block = source.read(8 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        written += len(block)
                        extracted += len(block)
                        if progress:
                            progress('install', extracted * 100 // total_runtime)
                if written != expected_size:
                    raise OSError(f'TensorRT runtime extraction was incomplete for {name}')
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in TENSORRT_DLLS:
            staging.joinpath(name).replace(target_dir / name)
        (target_dir / '.rfdetr-runtime.json').write_text(json.dumps({
            'version': TENSORRT_VERSION,
            'wheel_sha256': TENSORRT_WHEEL_SHA256,
        }, indent=2), encoding='utf-8')
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def ensure_runtime(progress=None):
    runtime = resolve_runtime_dir()
    if runtime is not None:
        if progress:
            progress('runtime', 100)
        return runtime
    if os.name != 'nt':
        raise RuntimeError('The built-in TensorRT RF-DETR runtime is available only on Windows')
    cache_root = get_config_dir() / f'rfdetr-runtime-{TENSORRT_VERSION}'
    target = cache_root / 'bin'
    wheel = cache_root / 'downloads' / TENSORRT_WHEEL_NAME
    _download(TENSORRT_WHEEL_URL, wheel, TENSORRT_WHEEL_SIZE,
              TENSORRT_WHEEL_SHA256, 'runtime', progress)
    _extract_runtime(wheel, target, progress)
    wheel.unlink(missing_ok=True)
    return target.resolve()


def ensure_resources(progress=None):
    model = ensure_model(progress)
    runtime = ensure_runtime(progress)
    return model, runtime


def download_progress(resource, percent):
    """CLI progress protocol consumed by the desktop's download control."""
    print(json.dumps({'type': 'resource-progress', 'resource': resource,
                      'percent': int(percent)}), flush=True)
