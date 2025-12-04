#!/usr/bin/env python3
"""
Phase 2 Demo: LiDAR-based 3D Object Detection (PointPillars)
Run inference on nuScenes data and print 3D coordinates.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from mmdet3d.apis import init_model, inference_detector

# Add project root directory to path to make sure we can load data.nuscenes_loader
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.nuscenes_loader import NuScenesLoader

def main():
    print("=" * 60)
    print("Phase 2: LiDAR 3D Detection Demo (PointPillars)")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Config path
    # ---------------------------------------------------------
    weights_dir = Path("models/weights")
    
    # Search .py config
    config_files = list(weights_dir.glob("*.py"))
    if not config_files:
        print("Error: Config file (.py) not found in models/weights/")
        return
    config_file = str(config_files[0])
    
    # Search .pth config
    checkpoint_files = list(weights_dir.glob("*.pth"))
    if not checkpoint_files:
        print("Error: Checkpoint file (.pth) not found in models/weights/")
        return
    checkpoint_file = str(checkpoint_files[0])

    print(f"Config:     {Path(config_file).name}")
    print(f"Checkpoint: {Path(checkpoint_file).name}")

    # ---------------------------------------------------------
    # 2. Model Initialization
    # ---------------------------------------------------------
    print("\nInitializing PointPillars model...")
    
    # For Mac users: forcibly use CPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        model = init_model(config_file, checkpoint_file, device=device)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("Tip: If using Mac, make sure mmdet3d is installed correctly.")
        return

    # ---------------------------------------------------------
    # 3. Data Loading
    # ---------------------------------------------------------
    print("\nLoading nuScenes dataset...")
    loader = NuScenesLoader(
        dataroot='./data/nuscenes',
        version='v1.0-mini',
        lidar_channel='LIDAR_TOP'
    )
    
    # Get first sample from the first scene
    scene = loader.get_scenes()[0]
    samples = loader.get_scene_samples(scene['token'])
    first_sample = samples[0]
    
    # Get LiDAR path
    data_dict = loader.get_multimodal_data(first_sample['token'])
    lidar_path = data_dict['lidar_path']
    print(f"Processing LiDAR file: {lidar_path}")

    # ---------------------------------------------------------
    # 4. Inference
    # ---------------------------------------------------------
    print("\nRunning inference (this might take a moment)...")
    
    # inference_detector receive models and LiDAR path
    # Read .bin file for pre-processing
    result = inference_detector(model, lidar_path)

    # ---------------------------------------------------------
    # 5. Result Analysis
    # ---------------------------------------------------------
    # MMDetection3D 1.x returns Det3DDataSample objects
    # We need to extract pred_instances_3d
    pred = result.pred_instances_3d
    
    # Get 3D boxes, scores and labels
    bboxes_3d = pred.bboxes_3d.tensor.numpy() if device == 'cpu' else pred.bboxes_3d.tensor.cpu().numpy()
    scores_3d = pred.scores_3d.numpy() if device == 'cpu' else pred.scores_3d.cpu().numpy()
    labels_3d = pred.labels_3d.numpy() if device == 'cpu' else pred.labels_3d.cpu().numpy()

    # Filter scores with low threshold (Score Thresh)
    score_thresh = 0.25
    mask = scores_3d > score_thresh
    
    final_boxes = bboxes_3d[mask]
    final_scores = scores_3d[mask]
    final_labels = labels_3d[mask]

    print(f"\nFound {len(final_boxes)} objects (score > {score_thresh}):")
    print("-" * 60)
    print(f"{'Class ID':<10} | {'Score':<8} | {'Location (x, y, z)':<30} | {'Dimensions (l, w, h)'}")
    print("-" * 60)

    for box, score, label in zip(final_boxes, final_scores, final_labels):
        # box format: [x, y, z, l, w, h, yaw, ...]
        loc = f"[{box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}]"
        dims = f"[{box[3]:.2f}, {box[4]:.2f}, {box[5]:.2f}]"
        print(f"{label:<10} | {score:.4f}   | {loc:<30} | {dims}")

    print("-" * 60)
    print("Step 4 Complete! The model is working.")
    print("Next Step: Project these 3D boxes onto the 2D image.")

if __name__ == "__main__":
    main()