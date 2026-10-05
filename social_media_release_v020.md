# 🚀 Social Media Launch Kit: LidarCamera360 v0.2.0

Banner Image to attach:
[`assets/release_v020_banner_3d.jpg`](file:///c:/Users/Everton-PC/Documents/APLICATIVOS/Lidar-camera-calibrator/assets/release_v020_banner_3d.jpg)

GitHub Release URL:
https://github.com/Everton-Braz/LidarCamera360/releases/tag/v0.2.0

Direct Portable Download:
https://github.com/Everton-Braz/LidarCamera360/releases/download/v0.2.0/LidarCamera360_portable.exe

---

## 👔 1. LinkedIn Post (Engineering & Geospatial Focus)

🚀 **LidarCamera360 v0.2.0 is officially released!** 🦅✨

We are thrilled to launch version **v0.2.0** of **LidarCamera360**, bringing major architectural upgrades, native hardware expansion, and breakthrough optimizations for 3D Gaussian Splatting (3DGS) and photogrammetry pipelines.

### 🌟 What's New in v0.2.0:

1️⃣ **3DMakerPro Eagle LiDAR Scanner Support** 🦅
Native support for the Eagle LiDAR scanner and Insta360 X6 cameras! Includes automated rigid lever-arm calibration, sensor sync, and multi-modal alignment.

2️⃣ **Breakthrough in 3D Gaussian Splatting (3DGS) Seeding** 🎯
- **Zero Downsampling Cap:** You can now feed 100% of your dense metric LiDAR point cloud (tested up to 28.5M+ points!) directly into modern 3DGS engines like Spirula Studio and LichtFeld Studio.
- **Faster Training Convergence:** High-density metric scaffolding dramatically speeds up training convergence on modern GPUs (tested with only ~6.5 GiB VRAM on an RTX 5070 Ti) by eliminating iterative densification lag.
- **Hybrid SfM + LiDAR Fusion:** For reconstruction workflows, the engine automatically fuses SfM camera ray tracks with dense metric LiDAR geometry into a unified COLMAP model.

3️⃣ **Instant COLMAP Export (Zero Duplicate Storage)** ⚡
Replaced legacy file duplicating with NTFS hardlinks and junctions. Exporting 3DGS datasets with hundreds of 3840x3840 fisheye frames now takes seconds and consumes 0 additional disk space!

4️⃣ **Timelapse & Variable-Rate Auto-Calibration** ⏱️
Direct binary telemetry decoding for INSV timelapses with adaptive IMU gyro cross-correlation for sub-millisecond sync.

5️⃣ **Interactive 3D OBB Clipping & Transform Gizmos** 📦
New 6-plane shader slicing box with real-time translation and rotation gizmos for fine sensor alignment.

📦 **Ready to use out-of-the-box!**
No Python environment or complicated setup required. Download the single-file Windows portable executable below:

🔗 **GitHub Release:** https://github.com/Everton-Braz/LidarCamera360/releases/tag/v0.2.0
📥 **Direct Portable Download:** https://github.com/Everton-Braz/LidarCamera360/releases/download/v0.2.0/LidarCamera360_portable.exe

#LiDAR #3DGaussianSplatting #Photogrammetry #EagleScanner #Insta360 #SpirulaStudio #LichtFeldStudio #SLAM #ComputerVision #Geospatial #OpenSource

---

## 👥 2. Facebook Post (Community & Visual Focus)

🦅 **Exciting News! LidarCamera360 v0.2.0 is HERE!** ✨

If you work with 3D scanners, 360° cameras, or 3D Gaussian Splatting, this release is for you! 🚀

### 🔥 What’s new in this version:
✅ **Full Eagle LiDAR Scanner & Insta360 X6 Support:** Plug-and-play workflows for the newest hardware on the market.
✅ **Full-Density 3D Gaussian Splatting:** No more downsampling! Feed all 28+ million points straight into Spirula Studio and LichtFeld Studio for sharper renders and faster training convergence.
✅ **Smart SfM + LiDAR Point Fusion:** Combines camera tracking points with laser-accurate LiDAR surfaces into one complete dataset.
✅ **Zero Extra Disk Space:** Instant exports using smart hardlinks—saving gigabytes of drive space!
✅ **Interactive 3D Clipping Box:** Inspect and slice your point clouds in real-time with 3D gizmos.

💻 **No installation required!**
Just download the portable `.exe` file, double-click, and start scanning and colorizing right away:

👉 **Download v0.2.0 on GitHub:**
https://github.com/Everton-Braz/LidarCamera360/releases/tag/v0.2.0

Let us know what you think and share your scans in the comments! 👇

#3DScanning #LiDAR #3DGS #GaussianSplatting #Insta360 #EagleScanner #3DReconstruction #TechNews

---

## 🐦 3. X / Twitter Post (Punchy Thread Format)

**Post 1 (Main Announcement):**
LidarCamera360 v0.2.0 is out! 🦅🚀

Full support for the Eagle LiDAR Scanner, native Insta360 X6 calibration, zero-downsampling 3DGS seeding (28M+ points in @SpirulaStudio), and automatic SfM + LiDAR point cloud fusion!

Download the portable Windows binary: 👇
https://github.com/Everton-Braz/LidarCamera360/releases/tag/v0.2.0

#LiDAR #3DGS #GaussianSplatting #EagleScanner #OpenSource

**Post 2 (Technical Details):**
Key upgrades in v0.2.0:
⚡ Full raw 28.5M+ point cloud seeds run on ~6.5GB VRAM & converge faster
⚡ Automatic SfM tie-point + LiDAR fusion preserving 2D camera tracks
⚡ NTFS hardlink exports (0 GB wasted storage)
⚡ Interactive 3D OBB clipping box & gizmos
⚡ INSV timelapse auto-sync
