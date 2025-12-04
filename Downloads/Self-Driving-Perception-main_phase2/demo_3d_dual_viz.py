#!/usr/bin/env python3
"""
3D Dual-View Visualization Demo (Debug Version)
Visualizes GROUND TRUTH 3D bounding boxes on Front AND Back camera images.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
DATAROOT = './data/nuscenes/v1.0-mini'
OUTPUT_DIR = './output'
VERSION = 'v1.0-mini'
FPS = 10.0

# Visualization config
FONT_SCALE = 0.8
FONT_THICKNESS = 2
BOX_THICKNESS = 3
# -------------------------------------------------------------------------

def render_3d_box_with_info(img, box, camera_intrinsic, color=(0, 255, 0)):
    """Project 3D box corners to 2D image plane, draw lines and DETAILED info."""
    if not box_in_image(box, camera_intrinsic, (img.shape[1], img.shape[0])):
        return img

    corners_3d = box.corners()
    corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]
    corners = corners_2d.astype(int).T

    # Draw Box Lines
    for i in range(4):
        cv2.line(img, tuple(corners[i]), tuple(corners[(i+1)%4]), color, BOX_THICKNESS)
        cv2.line(img, tuple(corners[i+4]), tuple(corners[(i+1)%4 + 4]), color, BOX_THICKNESS)
        cv2.line(img, tuple(corners[i]), tuple(corners[i+4]), color, BOX_THICKNESS)
    
    cv2.line(img, tuple(corners[0]), tuple(corners[5]), (0, 255, 255), 2)
    cv2.line(img, tuple(corners[1]), tuple(corners[4]), (0, 255, 255), 2)

    # Label Data
    dist = box.center[2]
    cls_name = box.name.split('.')[-1].upper()
    label_text = f"{cls_name} {dist:.1f}m"
    
    label_x = int(min([c[0] for c in corners]))
    label_y = int(min([c[1] for c in corners]))
    label_y = max(label_y, 30)

    (w, h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
    cv2.rectangle(img, (label_x, label_y - h - 10), (label_x + w + 10, label_y + 5), color, -1)
    
    text_color = (0, 0, 0)
    cv2.putText(img, label_text, (label_x + 5, label_y - 5), 
               cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, text_color, FONT_THICKNESS)

    return img

def process_camera_view(nusc, sample, cam_channel, title):
    """Process a single camera view."""
    try:
        # 1. Get Camera Data
        cam_token = sample['data'][cam_channel]
        cam_data = nusc.get('sample_data', cam_token)
        
        # 2. Load Image
        img_path = os.path.join(DATAROOT, cam_data['filename'])
        if not os.path.exists(img_path):
            print(f"⚠️ Warning: Image not found at {img_path}")
            return None
            
        img = cv2.imread(img_path)
        if img is None: 
            print(f"⚠️ Warning: Failed to decode image {img_path}")
            return None
        
        # 3. Get Calibration
        cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        camera_intrinsic = np.array(cs_record['camera_intrinsic'])
        
        # 4. Get Boxes
        _, boxes, _ = nusc.get_sample_data(cam_token, box_vis_level=BoxVisibility.ANY)
        
        # 5. Draw Boxes
        object_count = 0
        for box in boxes:
            if 'vehicle' not in box.name and 'human' not in box.name:
                continue

            if 'car' in box.name: color = (0, 255, 0)
            elif 'truck' in box.name: color = (0, 165, 255)
            elif 'bus' in box.name: color = (0, 255, 255)
            elif 'pedestrian' in box.name: color = (0, 100, 255)
            else: color = (200, 200, 200)
            
            img = render_3d_box_with_info(img, box, camera_intrinsic, color)
            object_count += 1
            
        # 6. Add Title
        cv2.rectangle(img, (0, 0), (img.shape[1], 80), (0, 0, 0), -1)
        cv2.putText(img, f"{title} | Objects: {object_count}", (30, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        return img
    except Exception as e:
        print(f"Error processing {cam_channel}: {e}")
        return None

def main():
    print("DEBUG: ----------------------------------------")
    print("DEBUG: Script initialized.")
    print(f"DEBUG: Current Working Directory: {os.getcwd()}")
    print(f"DEBUG: Looking for dataset at: {os.path.abspath(DATAROOT)}")
    
    if not os.path.exists(DATAROOT):
        print(f"❌ CRITICAL ERROR: Dataset directory not found!")
        print(f"   Expected: {os.path.abspath(DATAROOT)}")
        print("   Please check if 'data/nuscenes/v1.0-mini' exists.")
        return

    try:
        print("DEBUG: Initializing NuScenes (this might take a few seconds)...")
        nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
        print("DEBUG: NuScenes initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing NuScenes: {e}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_path = os.path.join(OUTPUT_DIR, '3d_dual_view_final.mp4')
    writer = None
    
    # Process first 3 scenes
    scenes_to_process = nusc.scene[:3]
    print(f"DEBUG: Found {len(nusc.scene)} scenes. Processing first {len(scenes_to_process)}...")
    
    total_frames = 0
    
    for i, scene in enumerate(scenes_to_process):
        print(f"DEBUG: Processing Scene {i+1}/{len(scenes_to_process)}: {scene['name']}")
        current_token = scene['first_sample_token']
        
        while current_token:
            sample = nusc.get('sample', current_token)
            
            # Process Views
            img_front = process_camera_view(nusc, sample, 'CAM_FRONT', "FRONT VIEW")
            img_back = process_camera_view(nusc, sample, 'CAM_BACK', "BACK VIEW")
            
            if img_front is not None and img_back is not None:
                # Vertical Stitch
                combined_img = cv2.vconcat([img_front, img_back])
                
                if writer is None:
                    h, w = combined_img.shape[:2]
                    print(f"DEBUG: Initializing Video Writer ({w}x{h} @ {FPS}fps)...")
                    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'avc1'), FPS, (w, h))
                
                writer.write(combined_img)
                total_frames += 1
                
                if total_frames % 20 == 0:
                    print(f"  -> Processed {total_frames} frames...", flush=True)
            else:
                print("⚠️ Warning: Skipped a frame due to loading error.")
            
            current_token = sample['next']

    if writer:
        writer.release()
        print(f"✅ SUCCESS: Video saved to {os.path.abspath(video_path)}")
        
        # try opening the video automatically
        try:
            import subprocess
            subprocess.run(["open", video_path], check=False)
        except:
            pass
    else:
        print("❌ Error: No frames were written to video.")

if __name__ == "__main__":
    main()