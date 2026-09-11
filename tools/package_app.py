"""Package the native estimator, app runtime and third-party notices."""
import argparse
import hashlib
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raven_app.branding import APP_NAME


def main():
    p = argparse.ArgumentParser(description=f"Package {APP_NAME} into standalone Windows binary")
    p.add_argument('--vcpkg-root', type=Path, default=None, help="Path to vcpkg root (optional)")
    p.add_argument('--onefile', action='store_true', help="Package as single-file portable executable")
    p.add_argument('--bundle-spirula', action='store_true', help='Compatibility flag; Spirula is always bundled')
    p.add_argument('--incremental', action='store_true', help='Reuse PyInstaller analysis cache')
    a = p.parse_args()

    # Validate required runtime modules are installed
    for module in ('PyQt6', 'numpy', 'scipy', 'cv2', 'av', 'qfluentwidgets'):
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin:
            raise SystemExit(f'{module} must be installed. Check requirements.txt.')

    native_dirs = [ROOT / 'build/native/Release', ROOT / 'bin']
    native = next((d for d in native_dirs if (d / 'fastlivo2.exe').is_file()), None)
    if native is None:
        raise SystemExit('Build the native engine first (fastlivo2.exe missing in build/ or bin/)')

    stage = ROOT / 'build/package-data'
    licenses = stage / 'licenses'
    licenses.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'FAST-LIVO2/LICENSE').is_file():
        shutil.copy2(ROOT / 'FAST-LIVO2/LICENSE', licenses / 'FAST-LIVO2.txt')

    for project in ('Sophus', 'rpg_vikit'):
        proj_dir = ROOT / 'build/deps-src' / project
        if proj_dir.is_dir():
            for f in proj_dir.rglob('*'):
                if f.is_file() and f.name.lower() in ('license', 'license.txt', 'copying'):
                    shutil.copy2(f, licenses / (project + '-' + str(f.relative_to(proj_dir)).replace('/', '_').replace('\\', '_')))

    if a.vcpkg_root and (a.vcpkg_root / 'installed/x64-windows/share').is_dir():
        for f in (a.vcpkg_root / 'installed/x64-windows/share').glob('*/copyright'):
            shutil.copy2(f, licenses / (f.parent.name + '.txt'))

    snapshots = {}
    for f in sorted((ROOT / 'FAST-LIVO2').rglob('*')):
        if f.is_file() and f.suffix in ('.cpp', '.h', '.yaml', '.txt'):
            snapshots[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()
    (stage / 'source-manifest.json').write_text(json.dumps(snapshots, indent=2))

    cmd = [
        sys.executable, '-m', 'PyInstaller', '--noconfirm',
        '--name', APP_NAME,
        '--onefile' if a.onefile else '--onedir',
        '--console',
        '--distpath', str(ROOT / 'dist/single-file' if a.onefile else ROOT / 'dist'),
        '--workpath', str(ROOT / 'build/pyinstaller'),
        '--specpath', str(ROOT / 'build'),
        '--paths', str(ROOT),
        '--collect-submodules', 'rosbags',
        '--collect-all', 'qfluentwidgets',
        '--collect-all', 'av',
        '--add-data', f'{ROOT / "FAST-LIVO2/config"};FAST-LIVO2/config',
        '--add-data', f'{ROOT / "calibracao_rigida_raven_insta360.json"};.',
        '--add-data', f'{stage};.',
    ]
    if (ROOT / 'assets/app.ico').is_file():
        cmd += ['--icon', str(ROOT / 'assets/app.ico')]
    if (ROOT / 'assets').is_dir():
        cmd += ['--add-data', f'{ROOT / "assets"};assets']
    if (ROOT / 'configs').is_dir():
        cmd += ['--add-data', f'{ROOT / "configs"};configs']
    if (ROOT / 'locales').is_dir():
        cmd += ['--add-data', f'{ROOT / "locales"};locales']
    if not a.incremental:
        cmd.append('--clean')
    # Conda-based Python keeps the stdlib SQLite runtime outside its DLLs folder.
    sqlite = Path(sys.base_prefix) / 'Library/bin/sqlite3.dll'
    if sqlite.is_file():
        cmd += ['--add-binary', f'{sqlite};.']
    if (ROOT / 'docs/STANDALONE.md').is_file():
        cmd += ['--add-data', f'{ROOT / "docs/STANDALONE.md"};docs']

    for name in ('torch', 'tensorflow', 'matplotlib', 'pandas', 'IPython', 'pytest', 'open3d', 'PyQt5', 'PySide6'):
        cmd += ['--exclude-module', name]

    added_binaries = set()
    def add_binary(src, dst='bin'):
        p = Path(src).resolve()
        if p not in added_binaries and p.is_file():
            added_binaries.add(p)
            cmd.extend(['--add-binary', f'{p};{dst}'])

    for f in sorted(native.glob('*')):
        if f.suffix.lower() in ('.exe', '.dll'):
            add_binary(f, 'bin')

    spirula = ROOT / 'spirula/spirula.exe'
    if not spirula.is_file():
        spirula = ROOT / 'bin/spirula.exe'
    if not spirula.is_file():
        raise SystemExit('Standalone build requires spirula/spirula.exe')
    add_binary(spirula, 'bin')
    for dependency in spirula.parent.glob('*.dll'):
        add_binary(dependency, 'bin')

    gpu_dirs = [ROOT / 'build/native/vulkan_colorizer/Release', ROOT / 'build/vulkan/Release', ROOT / 'bin']
    gpu = next((d for d in gpu_dirs if (d / 'vulkan_colorizer.exe').is_file()), None)
    if gpu is None:
        raise SystemExit('Build vulkan_colorizer before packaging')
    for binary in gpu.iterdir():
        if binary.suffix.lower() in ('.exe', '.dll'):
            add_binary(binary, 'bin')

    for shader in ('occlusion_zbuf', 'colorize_consensus', 'resolve_consensus'):
        source = gpu / 'shaders' / (shader + '.spv')
        if not source.is_file():
            source = ROOT / 'native/vulkan_colorizer/shaders' / (shader + '.spv')
        if not source.is_file():
            raise SystemExit(f'Missing compiled shader: {source}')
        cmd += ['--add-data', f'{source};bin/shaders']

    entry = ROOT / 'lidarcamera360.py' if (ROOT / 'lidarcamera360.py').is_file() else ROOT / 'raven.py'
    cmd.append(str(entry))
    print(f"[*] Running PyInstaller packaging ({'ONEFILE' if a.onefile else 'ONEDIR'})...")
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    if sys.platform == 'win32':
        paths = [
            str(Path(os.environ['SystemRoot']) / 'System32'),
            str(Path(sys.base_prefix)),
            str(Path(sys.base_prefix) / 'Scripts'),
            str(Path(sys.base_prefix) / 'Library/bin'),
            str(Path(sys.base_prefix) / 'Library/usr/bin'),
        ]
        existing = env.get('PATH', '')
        env['PATH'] = os.pathsep.join([p for p in paths if Path(p).is_dir()] + ([existing] if existing else []))
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)

    output = ROOT / 'dist/single-file' if a.onefile else ROOT / f'dist/{APP_NAME}'
    output.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'docs/STANDALONE.md').is_file():
        shutil.copy2(ROOT / 'docs/STANDALONE.md', output / 'README.md')
    # Also make the native engine runnable on its own, without relying on the
    # parent bootloader's DLL directory or a system Visual C++ installation.
    if a.onefile:
        portable_target = ROOT / f'dist/{APP_NAME}_portable.exe'
        shutil.copy2(output / f'{APP_NAME}.exe', portable_target)
        print(f"[+] Portable executable saved to: {portable_target}")
    else:
        internal = output / '_internal'
        bin_dir = internal / 'bin'
        if bin_dir.is_dir():
            for pattern in ('vcruntime*.dll', 'msvcp*.dll', 'vcomp*.dll', 'concrt*.dll'):
                for f in internal.glob(pattern):
                    shutil.copy2(f, bin_dir / f.name)
    print(output / f'{APP_NAME}.exe')


if __name__=='__main__':main()
