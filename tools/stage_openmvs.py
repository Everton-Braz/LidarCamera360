"""Copy a fisheye OpenMVS build into openmvs/ for packaging, and zip its source for the release.

    python tools/stage_openmvs.py <folder with src, make, triplets and vcpkg, see third_party/openmvs>
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'd235f83275a29a22e3c8ef0d91b6e15448c94aa5'
PROGRAMS = ('DensifyPointCloud.exe', 'ReconstructMesh.exe', 'TextureMesh.exe')
RUNTIME = ('msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll', 'msvcp140_atomic_wait.dll', 'vcruntime140.dll',
           'vcruntime140_1.dll', 'vcomp140.dll', 'concrt140.dll')
# The libraries in the programs whose licence (GPL, LGPL) asks for their source too.
COPYLEFT = ('CGAL-cgal', 'gmp-', 'strukturag-libheif')


def git(src, *args):
    return subprocess.run(['git', '-C', str(src), *args], check=True, capture_output=True, text=True, encoding='utf-8').stdout


def check_source(src):
    """The zip must be the source of the binary: the pinned commit plus the patch of this repository, nothing else."""
    patch = (ROOT / 'third_party/openmvs/fisheye.patch').read_text(encoding='utf-8')
    head = git(src, 'rev-parse', 'HEAD').strip()
    lines = lambda text: [x.rstrip() for x in text.splitlines() if not x.startswith('index ')]
    if head != COMMIT or lines(git(src, 'diff')) != lines(patch):
        raise SystemExit(f'{src} is not commit {COMMIT} with third_party/openmvs/fisheye.patch')


def vcpkg_commit(build):
    if (build / 'vcpkg/.git').exists():
        return git(build / 'vcpkg', 'rev-parse', 'HEAD').strip()
    with tarfile.open(build / 'vcpkg.tgz') as tar:  # A GitHub archive carries its commit as a comment.
        return tar.pax_headers['comment']


def runtime_dlls():
    root = os.environ.get('VCToolsRedistDir')
    roots = [Path(root)] if root else []
    for base in (os.environ['ProgramFiles'], os.environ.get('ProgramFiles(x86)', '')):
        roots += sorted(Path(base).glob('Microsoft Visual Studio/*/*/VC/Redist/MSVC/14.*'), reverse=True)
    for folder in roots:
        found = {f.name: f for f in folder.glob('x64/Microsoft.VC*.*/*.dll')}
        if all(name in found for name in RUNTIME):
            return [found[name] for name in RUNTIME]
    raise SystemExit('The Visual C++ runtime DLLs were not found; run this from a Visual Studio prompt')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('build', type=Path)
    p.add_argument('--source-zip', type=Path, default=ROOT / 'dist/OpenMVS-fisheye-source.zip')
    a = p.parse_args()
    src, make = a.build / 'src', a.build / 'make'
    check_source(src)
    release = make / 'bin/vc17/x64/Release'
    missing = [name for name in PROGRAMS if not (release / name).is_file()]
    if missing:
        raise SystemExit(f'Build these first: {", ".join(missing)}')
    commit, runtime = vcpkg_commit(a.build), runtime_dlls()
    share = make / 'vcpkg_installed/x64-windows-rel/share'
    ports = sorted(f.parent.name for f in share.glob('*/copyright'))

    out = ROOT / 'openmvs'
    if out.exists():
        shutil.rmtree(out)
    licenses = out / 'licenses'
    licenses.mkdir(parents=True)
    for f in [release / name for name in PROGRAMS] + sorted(release.glob('*.dll')) + runtime:
        shutil.copy2(f, out / f.name)
    shutil.copy2(src / 'LICENSE', licenses / 'OpenMVS-AGPL-3.0.txt')
    shutil.copy2(src / 'COPYRIGHT.md', licenses / 'OpenMVS-COPYRIGHT.md')
    texts = {}
    for port in ports:
        text = (share / port / 'copyright').read_text(encoding='utf-8', errors='replace').strip()
        texts.setdefault(text, []).append(port)
    line = '=' * 80
    (licenses / 'libraries.txt').write_text('\n\n'.join(
        f'{line}\n{", ".join(names)}\n{line}\n{text}' for text, names in texts.items()) + '\n', encoding='utf-8')
    shutil.copy2(ROOT / 'licenses/OpenMVS-SOURCE.md', licenses / 'SOURCE.md')

    tarballs = sorted(f for f in (a.build / 'vcpkg/downloads').iterdir() if f.is_file() and f.name.startswith(COPYLEFT))
    if len(tarballs) < len(COPYLEFT):
        raise SystemExit(f'Expected the sources of {COPYLEFT} in vcpkg/downloads, found {[f.name for f in tarballs]}')
    a.source_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.source_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in git(src, 'ls-files').splitlines():
            z.write(src / name, f'src/{name}')
        for f in sorted((ROOT / 'third_party/openmvs').iterdir()):
            z.write(f, f.name)
        for port in ports:
            folder = a.build / 'vcpkg/ports' / port
            if not folder.is_dir():
                folder = src / 'ports' / port
            for f in sorted(folder.rglob('*')):
                if f.is_file():
                    z.write(f, f'vcpkg-ports/{port}/{f.relative_to(folder).as_posix()}')
        for f in tarballs:
            z.write(f, f'libraries/{f.name}', zipfile.ZIP_STORED)
        z.writestr('VCPKG_COMMIT.txt', f'https://github.com/microsoft/vcpkg/tree/{commit}\n')
    print(f'{out}: {sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6:.0f} MB')
    print(f'{a.source_zip}: {a.source_zip.stat().st_size / 1e6:.0f} MB; attach it to the release')


if __name__ == '__main__':
    main()
