"""Package the native estimator, app runtime and third-party notices."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description="Package RavenCalibrator into standalone Windows binary")
    p.add_argument('--vcpkg-root', type=Path, default=None, help="Path to vcpkg root (optional)")
    p.add_argument('--onefile', action='store_true', help="Package as single-file portable executable")
    p.add_argument('--bundle-spirula', action='store_true', help="Bundle spirula.exe directly into package payload")
    a = p.parse_args()

    native = ROOT / 'build/native/Release'
    if not (native / 'fastlivo2.exe').is_file():
        raise SystemExit('Build the native engine first (build/native/Release/fastlivo2.exe missing)')

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
        sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
        '--name', 'RavenCalibrator',
        '--onefile' if a.onefile else '--onedir',
        '--console',
        '--distpath', str(ROOT / 'dist/single-file' if a.onefile else ROOT / 'dist'),
        '--workpath', str(ROOT / 'build/pyinstaller'),
        '--specpath', str(ROOT / 'build'),
        '--paths', str(ROOT),
        '--collect-submodules', 'rosbags',
        '--collect-all', 'qfluentwidgets',
        '--collect-all', 'PyQt6',
        '--add-data', f'{ROOT / "FAST-LIVO2/config"};FAST-LIVO2/config',
        '--add-data', f'{ROOT / "calibracao_rigida_raven_insta360.json"};.',
        '--add-data', f'{stage};.',
    ]
    if (ROOT / 'docs/STANDALONE.md').is_file():
        cmd += ['--add-data', f'{ROOT / "docs/STANDALONE.md"};docs']

    for name in ('torch', 'tensorflow', 'matplotlib', 'pandas', 'IPython', 'pytest', 'open3d', 'PyQt5', 'PySide6'):
        cmd += ['--exclude-module', name]

    for f in sorted(native.glob('*')):
        if f.suffix.lower() in ('.exe', '.dll'):
            cmd += ['--add-binary', f'{f};bin']

    spirula = ROOT / 'spirula/spirula.exe'
    if a.bundle_spirula and spirula.is_file():
        cmd += ['--add-binary', f'{spirula};spirula']

    cmd.append(str(ROOT / 'raven.py'))
    print(f"[*] Running PyInstaller packaging ({'ONEFILE' if a.onefile else 'ONEDIR'})...")
    subprocess.run(cmd, check=True, cwd=ROOT)

    output = ROOT / 'dist/single-file' if a.onefile else ROOT / 'dist/RavenCalibrator'
    output.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'docs/STANDALONE.md').is_file():
        shutil.copy2(ROOT / 'docs/STANDALONE.md', output / 'README.md')
    # Also make the native engine runnable on its own, without relying on the
    # parent bootloader's DLL directory or a system Visual C++ installation.
    if not a.onefile:
        internal=output/'_internal'
        for pattern in ('vcruntime*.dll','msvcp*.dll','vcomp*.dll','concrt*.dll'):
            for f in internal.glob(pattern):shutil.copy2(f,internal/'bin'/f.name)
    print(output/'RavenCalibrator.exe')


if __name__=='__main__':main()
