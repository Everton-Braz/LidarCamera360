import subprocess, requests, os, sys

# Obter token do git credential manager (wincred)
proc = subprocess.Popen(['git', '-c', 'credential.helper=', '-c', 'credential.helper=wincred', 'credential', 'fill'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
out, _ = proc.communicate('protocol=https\nhost=github.com\n\n')
creds = dict(line.split('=', 1) for line in out.strip().split('\n') if '=' in line)
token = creds.get('password')
if not token:
    print('Token not found in wincred!')
    sys.exit(1)

headers = {'Authorization': f'token {token}', 'Accept': 'application/vnd.github.v3+json'}
repo = 'Everton-Braz/LidarCamera360'

# Obter release existente pela tag v0.3.0
r = requests.get(f'https://api.github.com/repos/{repo}/releases/tags/v0.3.0', headers=headers)
if r.status_code != 200:
    print('Failed to get release by tag:', r.status_code, r.text)
    sys.exit(1)

rel = r.json()
rel_id = rel['id']
upload_url = rel['upload_url'].split('{')[0]
print(f'Found release ID {rel_id}. Existing assets: {[a["name"] for a in rel.get("assets", [])]}')

release_notes = """## What's Changed in v0.3.0

### Native 360° Equirectangular Video Support
* Hardware-accelerated decoding for 360° panoramic MP4 and dual-fisheye INSV video streams.
* Spherical camera projection (COLMAP Model 17) for single- and dual-lens panoramic setups.

### RootSLAM Global Pose Graph SLAM Optimization (Loop Closure)
* Integration of square-root bundle adjustment formulation (Santucci et al., arXiv:2610.15157).
* QR-based nullspace marginalization ($Q_2^T$) preventing condition number squaring.
* SE(3)/SO(3) manifold retractions in float32 arithmetic, eliminating long-corridor trajectory drift and ghosting.

### RF-DETR Person & Operator Masking via Vulkan
* Accelerated deep segmentation via Vulkan to detect and mask pedestrians.
* Operator cleanup radius (`--operator-radius`) to avoid dynamic ghosting in point clouds and 3DGS training datasets.

### Full-Density 3DGS Dataset Export
* Export sparse point cloud seeds directly to PostShot, Spirula Studio, and Nerfstudio without downsampling (tested up to 28.5M+ points).
* NTFS hardlink restructuring with zero disk storage overhead.

### Multi-Camera & Auxiliary Sensor Fusion
* Simultaneous calibration and fusion of a 3rd high-resolution forward-facing camera (smartphone/action-cam) alongside 360° rigs.

### Vulkan GPU Colorizer
* Multi-view top-3 consensus color projection powered by Vulkan compute shaders.

### Binaries Included
* `LidarCamera360-v0.3.0-windows-x64.zip`: Standalone directory bundle.
* `LidarCamera360_portable.exe`: Standalone single-file portable executable (no ROS, WSL, or Linux setup needed).
"""

# Atualizar body e title do release
r_patch = requests.patch(f'https://api.github.com/repos/{repo}/releases/{rel_id}', headers=headers, json={'name': 'LidarCamera360 v0.3.0', 'body': release_notes})
print('Updated release details:', r_patch.status_code)

assets = [
    ('dist/LidarCamera360_portable.exe', 'application/vnd.microsoft.portable-executable', 'LidarCamera360_portable.exe'),
    ('dist/LidarCamera360-v0.3.0-windows-x64.zip', 'application/zip', 'LidarCamera360-v0.3.0-windows-x64.zip')
]

existing_asset_names = {a['name']: a['id'] for a in rel.get('assets', [])}

for file_path, content_type, asset_name in assets:
    if asset_name in existing_asset_names:
        print(f'Asset {asset_name} already exists (ID: {existing_asset_names[asset_name]}). Deleting old one...')
        requests.delete(f'https://api.github.com/repos/{repo}/releases/assets/{existing_asset_names[asset_name]}', headers=headers)

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f'Uploading {asset_name} ({size_mb:.1f} MB)...')
    with open(file_path, 'rb') as f:
        up_headers = {
            'Authorization': f'token {token}',
            'Content-Type': content_type
        }
        res = requests.post(f'{upload_url}?name={asset_name}', headers=up_headers, data=f)
        if res.status_code in (200, 201):
            print(f'Uploaded {asset_name} successfully!')
        else:
            print(f'Failed to upload {asset_name}:', res.status_code, res.text)

print('Release update completed!')
