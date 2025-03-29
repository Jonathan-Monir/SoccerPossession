import os
import cv2
import numpy as np
import shutil
from ultralytics import YOLO
from norfair import Tracker, Video
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean
from tracking.inference.converter import Converter
import run_utils as ru

# Paths
video_path = "manc.mp4"
yolo_path = "yolo8.pt"  # Update with actual model path
fps = 30
frames_dir = 'frames'   # Update this path
labels_dir = 'labels'   # Update this path

# Ensure output directories exist
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

def delete_folder_contents(folder: str) -> None:
    """Deletes all files in the specified folder."""
    if not os.path.exists(folder):
        print(f"The folder '{folder}' does not exist.")
        return
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Failed to delete '{item_path}'. Reason: {e}")

def extract_frames(video_path, output_dir, fps):
    """Extracts frames from a video and saves them."""
    delete_folder_contents(output_dir)
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    saved_count = 0
    
    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % int(video_fps / fps) == 0:
            frame_path = os.path.join(output_dir, f'frame_{saved_count}.jpg')
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        frame_count += 1
    cap.release()

def process_video(yolo_path, video_path, fps, frames_dir, labels_dir):
    """Processes the video and saves detection results."""
    delete_folder_contents(frames_dir)
    delete_folder_contents(labels_dir)
    
    yolo_detector = YOLO(yolo_path)
    player_tracker = Tracker(distance_function=mean_euclidean, distance_threshold=250)
    ball_tracker = Tracker(distance_function=mean_euclidean, distance_threshold=150)
    motion_estimator = MotionEstimator()
    video = Video(input_path=video_path, output_fps=fps)
    
    for i, frame in enumerate(video):
        ball_detections = ru.get_detections(yolo_detector, frame, class_id=0, confidence_threshold=0.3)
        player_detections = ru.get_detections(yolo_detector, frame, class_id=1, confidence_threshold=0.35)
        
        # Combine detections and update motion estimation
        detections = ball_detections + player_detections
        coord_transformations = ru.update_motion_estimator(
            motion_estimator=motion_estimator, detections=detections, frame=frame
        )

        # Tracking: update trackers for players and ball separately
        player_track_objects = player_tracker.update(
            detections=player_detections, coord_transformations=coord_transformations
        )
        ball_track_objects = ball_tracker.update(
            detections=ball_detections, coord_transformations=coord_transformations
        )

        player_detections = Converter.TrackedObjects_to_Detections(player_track_objects)
        ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects)
        
        frame_path = os.path.join(frames_dir, f'frame_{i}.jpg')
        cv2.imwrite(frame_path, frame)
        
        label_path = os.path.join(labels_dir, f'frame_{i}.txt')
        with open(label_path, 'w') as f:
            for det in ball_detections:
                x, y, w, h = map(int, det.points.flatten())
                class_id = 0
                f.write(f"{class_id} {x} {y} {w} {h}\n")

            for det in player_detections:
                x, y, w, h = map(int, det.points.flatten())
                class_id = 1
                f.write(f"{class_id} {x} {y} {w} {h}\n")


        
    print("Processing complete. Frames and labels saved.")

# Run processing
process_video(yolo_path, video_path, fps, frames_dir, labels_dir)
