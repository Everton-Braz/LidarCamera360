# LidarCamera360 v0.3.0 Social Media Announcements

**Banner Image:** `assets/release_v0.3.0_banner.jpg` (1678x937, 16:9 ratio)

---

## 💼 LinkedIn Post

🚀 **Introducing LidarCamera360 v0.3.0 — Next-Gen LiDAR + 360° Sensor Fusion & RootSLAM Loop Closure on Native Windows!**

Building high-precision, colorized 3D point clouds and 3D Gaussian Splatting (3DGS) scenes from handheld LiDAR and consumer 360° cameras shouldn't require complex Linux VM setups, ROS workspaces, or lossy point cloud downsampling.

Today, we are excited to release **LidarCamera360 v0.3.0**, a major engineering milestone bringing state-of-the-art numerical optimization and panoramic vision directly to a standalone Windows executable.

🔥 **What’s New in v0.3.0:**

🔹 **RootSLAM-Powered Global Loop Closure:** 
We’ve integrated a cutting-edge **Square-Root Bundle Adjustment (RootSLAM)** solver (*Santucci et al., arXiv:2610.15157*). By projecting landmark and plane blocks onto the nullspace $Q_2^T$ rather than forming the normal equations ($J^T J$), we avoid squaring the condition number. The result? Rock-solid `float32` numerical stability, manifold $SE(3)/SO(3)$ updates, and zero trajectory drift, doubled walls, or ghosting across long indoor/outdoor revisits.

🔹 **Native 360° Equirectangular Video Support:**
Hardware-accelerated decoding for stitched panoramic `.mp4` and dual-fisheye `.insv` streams, complete with native Spherical Camera Projection (COLMAP Model 17) for both single- and dual-lens rigs.

🔹 **Full-Density 3DGS Seed Initialization (28.5M+ Points):**
Export directly to **Spirula Studio, PostShot, and Nerfstudio** with zero downsampling—tested at over **28.5 million dense LiDAR points**! Smart NTFS hardlinking restructures training datasets instantaneously with **0 bytes** of extra disk overhead.

🔹 **Multi-Camera & Auxiliary Sensor Arrays:**
Simultaneously calibrate and fuse a 3rd high-res forward camera (smartphone or action-cam) alongside multiple 360° rigs.

🔹 **Vulkan GPU Compute Colorizer:**
Blazingly fast multi-view top-3 consensus color projection powered by native Vulkan compute shaders.

🔹 **Zero ROS. Zero Linux. 100% Native Windows:**
Available as both an installer/directory bundle and a single-file portable executable (`LidarCamera360_portable.exe`) for Windows 10/11.

🔗 **Download v0.3.0 & Explore the Source on GitHub:**
https://github.com/Everton-Braz/LidarCamera360/releases/tag/v0.3.0

#LidarCamera360 #ComputerVision #3DGaussianSplatting #LiDAR #SLAM #RootSLAM #Photogrammetry #Robotics #PointCloud #Vulkan #3DReconstruction #OpenSource #DigitalTwins

---

## 𝕏 X (Twitter) Thread

**Tweet 1/5 (Main Post + Banner Image):**
🚀 **LidarCamera360 v0.3.0 is LIVE!**

Fuse LiDAR + 360° video into ultra-dense colorized point clouds & 3D Gaussian Splatting scenes—100% native on Windows (Zero ROS, Zero Linux!).

Featuring **RootSLAM** loop closure, **28.5M+ pt** 3DGS exports, & Vulkan GPU compute! 🧵👇

🔗 https://github.com/Everton-Braz/LidarCamera360/releases/tag/v0.3.0

**Tweet 2/5 (RootSLAM Math & Loop Closure):**
1️⃣ **RootSLAM Global Pose Graph & Loop Closure** 📐
Say goodbye to trajectory drift and doubled walls! 
v0.3.0 implements Square-Root Bundle Adjustment (*Santucci et al., 2026*). Using $Q_2^T$ nullspace marginalization on $SE(3)/SO(3)$ manifolds, it delivers extreme numerical stability in fast `float32` without squaring the condition number.

**Tweet 3/5 (360° Video & Multi-Cam):**
2️⃣ **Native 360° Equirectangular + Multi-Camera Fusion** 🌐📸
• Hardware-accelerated `.mp4` panoramic & `.insv` dual-fisheye decoding
• Native COLMAP Model 17 spherical projection
• Fuse a 3rd auxiliary forward camera (smartphone/action-cam) + multiple 360° rigs simultaneously!

**Tweet 4/5 (3DGS & Vulkan):**
3️⃣ **28.5M+ Point 3DGS Initialization & Vulkan Colorizer** ⚡
• Export full-density LiDAR seeds to **PostShot, Spirula Studio, & Nerfstudio** with ZERO downsampling.
• NTFS hardlinks create training-ready folders using **0 bytes** of extra disk space.
• Multi-view top-3 consensus colorization via native **Vulkan GPU Compute Shaders**.

**Tweet 5/5 (CTA):**
4️⃣ **Grab the Single-File Portable EXE** 💻
No Docker, no WSL, no ROS headaches. Just download `LidarCamera360_portable.exe` on Windows 10/11 and start scanning!

⭐ Star the repo & download v0.3.0 here:
https://github.com/Everton-Braz/LidarCamera360

#3DGS #GaussianSplatting #LiDAR #SLAM #ComputerVision #Vulkan

---

## 📘 Facebook Post

🎉 **Big Release Alert: LidarCamera360 v0.3.0 is Out Now!** 🎉

If you work with 3D Gaussian Splatting (3DGS), laser scanning, photogrammetry, or robotics, our biggest update yet makes fusing handheld LiDAR with 360° cameras faster, sharper, and easier than ever—running **100% natively on Windows 10/11 with ZERO ROS or Linux required!**

✨ **Highlights of v0.3.0:**
✅ **Native 360° Equirectangular Video:** Direct hardware decoding for stitched 360° `.mp4` and `.insv` videos with true spherical projection (COLMAP Model 17).
✅ **RootSLAM Loop Closure:** Powered by Square-Root Bundle Adjustment (*Santucci et al., 2026*), eliminating long-corridor drift, ghosting, and double walls with high-speed `float32` precision.
✅ **Full-Density 3DGS Exports (28.5M+ Points!):** Send massive, un-downsampled LiDAR point clouds straight into **PostShot, Spirula Studio, or Nerfstudio**. Uses smart NTFS hardlinks so dataset export takes **0 bytes** of extra storage!
✅ **Multi-Camera Arrays:** Add a 3rd high-detail front camera (phone or action cam) alongside your 360° rig for ultra-crisp foreground textures.
✅ **Vulkan GPU Colorizer:** Lightning-fast multi-view color consensus powered by your GPU.
✅ **Plug & Play:** Available as a standalone portable executable (`LidarCamera360_portable.exe`).

📥 **Download the free release and check out the full changelog on GitHub:**
👉 https://github.com/Everton-Braz/LidarCamera360/releases/tag/v0.3.0

Drop a comment below with your scans or let us know what sensor rig you're running! 🚀👇

#3DGaussianSplatting #LiDAR #3DScanning #Photogrammetry #PointCloud #LidarCamera360 #ComputerVision #3DModeling
