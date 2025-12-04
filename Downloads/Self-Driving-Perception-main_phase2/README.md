# Self-Driving Perception System

A phased exploration of autonomous driving perception, evolving from 2D multi-view object detection to SOTA 3D LiDAR-based perception using the **nuScenes** dataset.

## Project Status
- ✅ Phase 1 Completed: Dual-View 2D Detection (Front + Back)  
- 🚧 Phase 2 In Progress: 3D LiDAR Inference & Visualization

## Project Overview
- **Objective:** Build a perception stack that processes multi-modal sensor data (Camera + LiDAR) for autonomous driving.  
- **Dataset:** mini-nuScenes v1.0-mini (6 Cameras, 1 LiDAR, Radar).  
- **Models:**
  - Phase 1: YOLOv8 (2D Object Detection)
  - Phase 2: PointPillars (3D LiDAR Detection via MMDetection3D)
- **Output:** Visualization videos demonstrating 2D & 3D situational awareness.

## Features

### Phase 1: 2D Dual-View Perception
- Split-Screen View: CAM_FRONT + CAM_BACK stitched for 360-degree awareness
- Real-time Detection: YOLOv8 inference on multiple camera feeds
- Distance Estimation: Heuristic depth estimation based on bounding box geometry
- Safety Warnings: color-coded boxes (Red: <15m, Orange: 15–30m, Green: >30m)

### Phase 2: 3D LiDAR Perception
- 3D Ground Truth Visualization: Projecting 3D LiDAR annotations onto 2D camera images
- Spatial Awareness: Visualizing 3D location `(x, y, z)`, dimensions `(l, w, h)`, and orientation
- LiDAR Inference: PointPillars model inference pipeline (Cloud/GPU recommended)
- Rendering: Produces enhanced dual-view 3D visualization videos

## Quick Start

### 1. Setup Environment

#### Option A — Phase 1 Environment (Local)
Suitable for running `video_demo.py` and `demo_3d_dual_viz.py`.

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B — Phase 2 Environment (GPU / Linux / Colab)
Required for running `demo_phase2_lidar.py`.

MMDetection3D is difficult to compile on macOS — it is recommended to use **Google Colab**, **Linux**, or any machine with **NVIDIA GPU support**.

---

### 2. Download Dataset

Download **mini-nuScenes v1.0-mini (~4GB)** from:  
https://www.nuscenes.org/download

Choose:  
**Full dataset (v1.0) → mini → `v1.0-mini.tgz`**

Extract to:
./data/nuscenes/

Verify that this path exists:
./data/nuscenes/v1.0-mini/maps

---

## 3. Run Demos

### Demo 1 — Phase 1: 2D Dual-View Detection

```bash
source venv/bin/activate
python video_demo.py
Output: output/dual_view_detection.mp4
```

### Demo 2 — Phase 2: 3D Ground Truth Visualization

```bash
source venv/bin/activate
python demo_3d_dual_viz.py
Output: output/3d_dual_view_enhanced.mp4
```

### Demo 3 — Phase 2: LiDAR Inference (PointPillars)

Requires MMDetection3D and GPU:

python demo_phase2_lidar.py

## Project Structure

```
self_driving_perception/
├── video_demo.py               # [Phase 1] Dual-View 2D Detection (YOLO)
├── demo_3d_dual_viz.py         # [Phase 2] 3D GT Visualization
├── demo_phase2_lidar.py        # [Phase 2] LiDAR Inference (PointPillars)
├── data/
│   ├── nuscenes_loader.py      # Enhanced loader: Camera + LiDAR + Calibration
│   └── nuscenes/v1.0-mini/     # Dataset location
├── models/
│   └── weights/                # Model weights (.pth) + configs (.py)
├── output/                     # Generated demo videos
├── requirements.txt            # Python dependencies for Phase 1
└── README.md                   # This file
```

## Technical Details

### Phase 1: 2D Pipeline
- **Inputs:** Synchronized `CAM_FRONT` and `CAM_BACK` images  
- **Detection:** YOLOv8n (runs per frame)  
- **Fusion:** Vertical concatenation using `cv2.vconcat` for a split-screen dashboard  
- **Depth Estimation:** Heuristic estimation using bounding-box height and pinhole camera geometry  

---

### Phase 2: 3D Pipeline

**Coordinate Transformation:**
LiDAR Frame (x, y, z) → Ego Frame → Camera Frame → Image Plane (u, v)

**Projection:**  
3D cuboid corners projected into 2D using the camera intrinsic matrix.

**Data Fusion:**  
Overlay LiDAR-derived spatial data (3D boxes, orientation, coordinates) onto camera RGB frames.

---

## Dependencies

### Core
- Python 3.8+
- PyTorch  
- OpenCV (`cv2`)  
- NumPy  
- nuscenes-devkit  
- ultralytics (YOLOv8)

### Optional (Phase 2)
- MMDetection3D  
- MMCV  
- CUDA Toolkit (for GPU acceleration)

---

## Troubleshooting

- **"Dataset not found":**  
  Ensure the directory exists:  

data/nuscenes/v1.0-mini/

- **"Video format not supported" on macOS:**  
Videos use H.264 (`avc1`). Try **VLC** if default macOS apps fail.

- **"No module named mmdet3d":**  
You are running a Phase 2 script without an MMDetection3D environment.  
Use **Google Colab** or a **Linux GPU machine**.

---

## License
MIT License — Educational project for autonomous driving perception.