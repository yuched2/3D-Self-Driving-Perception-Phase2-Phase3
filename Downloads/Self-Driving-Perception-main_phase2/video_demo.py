#!/usr/bin/env python3
"""
Video Demo for Self-Driving Perception System
Generates a split-screen video (Front + Back) with bounding boxes around detected cars.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from ultralytics import YOLO
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class VideoPerceptionDemo:
    """Generate video demo with YOLOv8 car detection on nuScenes dataset."""
    
    def __init__(
        self,
        data_root: str = "./data/nuscenes/v1.0-mini",
        output_dir: str = "./output",
        model_size: str = "n"  # n, s, m, l, x (nano to extra-large)
    ):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load YOLOv8 model
        print(f"Loading YOLOv8{model_size} model...")
        self.model = YOLO(f"yolov8{model_size}.pt")
        print("✓ Model loaded")
        
        # COCO relevant class IDs
        self.detection_classes = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            5: "bus", 7: "truck", 9: "traffic light", 11: "stop sign"
        }
        
        # Colors (BGR)
        self.colors = {
            0: (255, 100, 100), 1: (255, 255, 0), 2: (0, 255, 0),
            3: (255, 0, 255), 5: (255, 165, 0), 7: (0, 165, 255),
            9: (0, 255, 255), 11: (0, 0, 255)
        }
        
        self.focal_length = 1266.0
        self.avg_car_height = 1.5
        
    def estimate_distance(self, bbox_width, bbox_height, img_width):
        if bbox_height > 0:
            return (self.avg_car_height * self.focal_length) / bbox_height
        return 50.0
    
    def get_distance_color(self, distance):
        if distance < 15: return (0, 0, 255)
        elif distance < 30: return (0, 165, 255)
        else: return (0, 255, 0)
    
    def process_frame(self, frame: np.ndarray, conf_threshold: float = 0.5, show_distance: bool = True) -> np.ndarray:
        """Process a single frame with YOLOv8 detection."""
        results = self.model(frame, verbose=False)[0]
        img_height, img_width = frame.shape[:2]
        annotated_frame = frame.copy()
        car_count = 0
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id in self.detection_classes and conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_height = y2 - y1
                distance = self.estimate_distance(x2-x1, bbox_height, img_width)
                
                color = self.get_distance_color(distance) if show_distance else self.colors.get(cls_id, (255, 255, 255))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                class_name = self.detection_classes[cls_id]
                label = f"{class_name} {conf:.2f}"
                if show_distance: label += f" | {distance:.1f}m"
                
                # Draw label
                font = cv2.FONT_HERSHEY_SIMPLEX
                (w, h), b = cv2.getTextSize(label, font, 0.6, 2)
                label_y = y1 - 10 if y1 > 30 else y2 + h + 10
                cv2.rectangle(annotated_frame, (x1, label_y - h - b), (x1 + w, label_y + b), color, -1)
                cv2.putText(annotated_frame, label, (x1, label_y - b), font, 0.6, (255, 255, 255), 2)
                car_count += 1
        
        return annotated_frame, car_count

    def create_video_from_frames(
        self,
        frames: List[np.ndarray],
        output_name: str = "demo_video.mp4",
        fps: float = 10.0
    ) -> Path:
        """Create video from list of processed numpy frames."""
        if not frames:
            raise ValueError("No frames provided")
            
        output_path = self.output_dir / output_name
        height, width = frames[0].shape[:2]
        
        video_writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        print(f"Writing video to {output_path}...")
        for i, frame in enumerate(frames):
            video_writer.write(frame)
            if (i+1) % 50 == 0:
                print(f"  Written {i+1}/{len(frames)} frames")
                
        video_writer.release()
        print(f"✓ Video saved: {output_path}")
        return output_path


def main():
    print("=" * 60)
    print("Self-Driving Perception - Dual View Demo (Front + Back)")
    print("=" * 60)
    
    demo = VideoPerceptionDemo(
        data_root="./data/nuscenes/v1.0-mini",
        output_dir="./output",
        model_size="n"
    )
    
    dataset_path = Path("./data/nuscenes/v1.0-mini")
    if not dataset_path.exists():
        print("⚠ nuScenes dataset not found! Please check path.")
        return

    try:
        from nuscenes.nuscenes import NuScenes
        print("Loading nuScenes dataset...")
        nusc = NuScenes(version='v1.0-mini', dataroot=str(dataset_path), verbose=False)
        
        # Process fewer scenes to save memory, since we store frames in RAM
        num_scenes = min(len(nusc.scene), 5) 
        print(f"Processing {num_scenes} scenes...")
        
        processed_frames = []
        
        for idx, scene in enumerate(nusc.scene[:num_scenes]):
            scene_name = scene['name']
            print(f"[{idx+1}/{num_scenes}] Processing {scene_name}...")
            
            sample_token = scene['first_sample_token']
            
            while sample_token:
                sample = nusc.get('sample', sample_token)
                
                # 1. Get Tokens
                cam_front_token = sample['data']['CAM_FRONT']
                cam_back_token = sample['data']['CAM_BACK']
                
                # 2. Get Paths
                front_data = nusc.get('sample_data', cam_front_token)
                back_data = nusc.get('sample_data', cam_back_token)
                
                path_front = dataset_path / front_data['filename']
                path_back = dataset_path / back_data['filename']
                
                # 3. Read Images
                img_front = cv2.imread(str(path_front))
                img_back = cv2.imread(str(path_back))
                
                if img_front is not None and img_back is not None:
                    # 4. Process Both
                    res_front, count_f = demo.process_frame(img_front, conf_threshold=0.5)
                    res_back, count_b = demo.process_frame(img_back, conf_threshold=0.5)
                    
                    # 5. Add Titles
                    cv2.putText(res_front, f"FRONT VIEW | Obj: {count_f}", (30, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    cv2.putText(res_back, f"BACK VIEW | Obj: {count_b}", (30, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    
                    # 6. Stack Images Vertically (Up/Down)
                    combined = cv2.vconcat([res_front, res_back])
                    
                    # Resize to prevent video being too huge (Optional, usually 1600x900)
                    # combined = cv2.resize(combined, (0,0), fx=0.5, fy=0.5)
                    
                    processed_frames.append(combined)
                
                sample_token = sample['next']
                
            print(f"  Current total frames: {len(processed_frames)}")
        
        # Create Video
        if processed_frames:
            video_path = demo.create_video_from_frames(
                processed_frames,
                output_name="dual_view_detection.mp4",
                fps=10.0
            )
            
            # Try to open
            import subprocess
            try:
                subprocess.run(["open", str(video_path)], check=False)
            except:
                pass
        else:
            print("No frames processed!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()