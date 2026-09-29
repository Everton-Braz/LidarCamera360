"""Lazy, verified RF-DETR resources for TensorRT and Vulkan mask backends."""
import hashlib
import importlib.util
import json
import os
from functools import lru_cache
from pathlib import Path
import shutil
import sys
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
VULKAN_EXECUTABLE = 'rfdetr-vulkan.exe'
VULKAN_SHADERS = ('ops.comp.spv', 'matmul.comp.spv', 'reduce.comp.spv')


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=128)
def _cached_sha256(path, size, modified_ns, created_ns):
    return _sha256(path)


def _file_sha256(path):
    path = Path(path)
    stat = path.stat()
    return _cached_sha256(str(path.resolve()), stat.st_size,
                          stat.st_mtime_ns, stat.st_ctime_ns)


def _valid_file(path, size, digest):
    path = Path(path)
    try:
        return path.is_file() and path.stat().st_size == size and _file_sha256(path) == digest
    except OSError:
        return False


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
    try:
        root = get_app_root()
        candidates = (get_config_dir() / 'models' / MODEL_NAME, root / 'models' / MODEL_NAME)
    except OSError:
        candidates = (get_app_root() / 'models' / MODEL_NAME,)
    for path in candidates:
        try:
            if path.is_file():
                return path.resolve()
        except OSError:
            continue
    return None


def model_is_ready(model_path=None):
    """Whether the selected/default model is present and passes its integrity check."""
    try:
        model = resolve_model(model_path)
        if model is None:
            return False
        if model_path is not None or os.environ.get('RAVEN_RFDETR_MODEL', '').strip():
            return model.stat().st_size > 0
        return _valid_file(model, MODEL_SIZE, MODEL_SHA256)
    except (OSError, ValueError):
        return False


def resolve_masker_executable():
    root = get_app_root()
    candidates = (root / 'bin/rfdetr-masker.exe',
                  root / 'build/rfdetr-masker/Release/rfdetr-masker.exe')
    return next((path.resolve() for path in candidates if path.is_file()), None)


def resolve_vulkan_executable():
    """Find the standalone Vulkan masker shipped in the app or development tree."""
    root = get_app_root()
    candidates = (root / 'bin' / VULKAN_EXECUTABLE,
                  root / 'build/vulkan-rfdetr/Release' / VULKAN_EXECUTABLE,
                  root / 'build/vulkan-rfdetr' / VULKAN_EXECUTABLE)
    return next((path.resolve() for path in candidates if path.is_file()), None)


def resolve_vulkan_shaders_dir(executable=None):
    """Resolve the compiled shader directory for a bundled or development runner."""
    executable = Path(executable).resolve() if executable else resolve_vulkan_executable()
    if executable is None:
        return None
    candidates = (executable.parent / 'shaders', executable.parent.parent)
    for directory in candidates:
        if all((directory / name).is_file() for name in VULKAN_SHADERS):
            return directory.resolve()
    return None


def _model_digest(model):
    if Path(model).name == MODEL_NAME and _valid_file(model, MODEL_SIZE, MODEL_SHA256):
        return MODEL_SHA256
    return _file_sha256(model)


def _schedule_path(model):
    fingerprint = _model_digest(model)
    cache = get_config_dir() / 'models' / 'vulkan'
    return cache / f'rfdetr-{fingerprint[:16]}.rvk', fingerprint


def resolve_vulkan_schedule(model_path=None):
    """Return a cached RVK schedule only when its model and file hashes match."""
    try:
        model = resolve_model(model_path)
        if model is None:
            return None
        schedule, fingerprint = _schedule_path(model)
        metadata_path = schedule.with_suffix(schedule.suffix + '.json')
        if not schedule.is_file() or not metadata_path.is_file():
            return None
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if (metadata.get('model_sha256') != fingerprint
                or metadata.get('schedule_bytes') != schedule.stat().st_size
                or metadata.get('schedule_sha256') != _file_sha256(schedule)):
            return None
        return schedule.resolve()
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def resource_status(backend='vulkan'):
    """Describe the first missing resource for the chosen backend."""
    if not model_is_ready():
        return 'model-missing'
    if backend == 'tensorrt':
        if resolve_masker_executable() is None:
            return 'executor-missing'
        return 'ready' if resolve_runtime_dir() is not None else 'runtime-missing'
    if backend == 'vulkan':
        executable = resolve_vulkan_executable()
        if executable is None or resolve_vulkan_shaders_dir(executable) is None:
            return 'executor-missing'
        return 'ready' if resolve_vulkan_schedule() is not None else 'schedule-missing'
    raise ValueError(f'Unknown RF-DETR backend: {backend}')


def resources_ready(backend='vulkan'):
    """Check all files required by the chosen mask backend without downloading."""
    return resource_status(backend) == 'ready'


def resolve_runtime_dir():
    """Find a complete TensorRT runtime, honoring an optional user override."""
    root = get_app_root()
    candidates = []
    override = os.environ.get('RAVEN_RFDETR_RUNTIME', '').strip()
    if override:
        path = Path(override).expanduser()
        candidates.extend((path, path / 'bin'))
    candidates.extend((root / 'bin', root / 'build/rfdetr-masker/Release'))
    try:
        candidates.append(get_config_dir() / f'rfdetr-runtime-{TENSORRT_VERSION}' / 'bin')
    except OSError:
        pass
    for directory in candidates:
        try:
            if all((directory / name).is_file()
                   and (directory / name).stat().st_size == size
                   for name, size in TENSORRT_DLLS.items()):
                return directory.resolve()
        except OSError:
            continue
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


def ensure_vulkan_schedule(model_path=None, progress=None):
    """Create and cache a static Vulkan schedule from the verified ONNX model."""
    model = ensure_model(progress) if model_path is None else resolve_model(model_path)
    executable = resolve_vulkan_executable()
    shaders = resolve_vulkan_shaders_dir(executable)
    if executable is None or shaders is None:
        raise RuntimeError('RF-DETR Vulkan executor or compiled shaders are missing from this app build')
    cached = resolve_vulkan_schedule(model)
    if cached is not None:
        if progress:
            progress('vulkan', 100)
        return cached

    schedule, fingerprint = _schedule_path(model)
    schedule.parent.mkdir(parents=True, exist_ok=True)
    lock = schedule.with_name(schedule.name + '.lock')
    deadline = time.monotonic() + 6 * 60 * 60
    descriptor = None
    while descriptor is None:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if not lock.exists() or time.time() - lock.stat().st_mtime > 6 * 60 * 60:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError('Another RF-DETR Vulkan schedule build is still active')
            time.sleep(1)
            cached = resolve_vulkan_schedule(model)
            if cached is not None:
                if progress:
                    progress('vulkan', 100)
                return cached
    os.write(descriptor, str(os.getpid()).encode('ascii'))
    os.close(descriptor)
    try:
        return _build_vulkan_schedule(model, schedule, fingerprint, progress)
    finally:
        lock.unlink(missing_ok=True)


def _build_vulkan_schedule(model, schedule, fingerprint, progress=None):
    temporary = schedule.with_name(schedule.stem + '.building.rvk')
    temporary_manifest = temporary.with_suffix(temporary.suffix + '.json')
    temporary.unlink(missing_ok=True)
    temporary_manifest.unlink(missing_ok=True)
    if progress:
        progress('vulkan', 1)
    exporter = get_app_root() / 'tools' / 'export_rfdetr_vulkan.py'
    if not exporter.is_file():
        raise RuntimeError('RF-DETR Vulkan model exporter is missing from this app build')
    spec = importlib.util.spec_from_file_location('_rfdetr_vulkan_exporter', exporter)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load RF-DETR Vulkan model exporter: {exporter}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        if progress:
            progress('vulkan', 3)
        spec.loader.exec_module(module)
        if progress:
            progress('vulkan', 7)
        module.export_model(
            model, temporary,
            progress=lambda percent: progress('vulkan', percent) if progress else None,
        )
        metadata = json.loads(temporary_manifest.read_text(encoding='utf-8'))
        if metadata.get('model_sha256') != fingerprint:
            raise RuntimeError('Vulkan schedule was exported from a different RF-DETR model')
        metadata['schedule_sha256'] = _file_sha256(temporary)
        temporary.replace(schedule)
        temporary_manifest.replace(schedule.with_suffix(schedule.suffix + '.json'))
        sidecar = schedule.with_suffix(schedule.suffix + '.json')
        metadata['schedule'] = str(schedule)
        sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    except Exception:
        temporary.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise
    finally:
        sys.modules.pop(spec.name, None)
    if progress:
        progress('vulkan', 100)
    return schedule.resolve()


def ensure_resources(progress=None, backend='vulkan', model_path=None):
    model = ensure_model(progress) if model_path is None else resolve_model(model_path)
    if backend == 'tensorrt':
        return model, ensure_runtime(progress)
    if backend == 'vulkan':
        return model, ensure_vulkan_schedule(model, progress)
    raise ValueError(f'Unknown RF-DETR backend: {backend}')


def download_progress(resource, percent):
    """CLI progress protocol consumed by the desktop's download control."""
    print(json.dumps({'type': 'resource-progress', 'resource': resource,
                      'percent': int(percent)}), flush=True)
