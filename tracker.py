import os
import cv2
import numpy as np
import PIL
from ultralytics import YOLO
from norfair import Tracker
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean
from preprosses import compute_noise, apply_nlm_denoising
import torch
from norfair import Tracker, Video
from tracking.inference.converter import Converter
# from tracking.inference import Converter
from tracking.soccer import Match, Player, Team
from tracking.soccer.draw import AbsolutePath
# from tracking.soccer.pass_event import Pass
import run_utils as ru

# Video and model paths
video_path = "manc.mp4"
fps = 10  # Target FPS for extraction


"""# Helpful functions"""


def delete_file(file_path):
    """
    Deletes the file at the specified file_path.

    Args:
        file_path (str): The path to the file to be deleted.
    """
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"An error occurred while deleting the file: {e}")
    else:
        print(f"File not found: {file_path}")


def process_video(yolo_path, video_path, fps):
    coord_transformations = []
    motion_estimators = []
    # Initializ YOLO detector with the given model path
    yolo_detector = YOLO(yolo_path)

    # Initialize trackers and motion estimator
    player_tracker = Tracker(distance_function=mean_euclidean, distance_threshold=250, initialization_delay=3, hit_counter_max=90)
    ball_tracker = Tracker(distance_function=mean_euclidean, distance_threshold=150, initialization_delay=20, hit_counter_max=2000)
    motion_estimator = MotionEstimator()
    coord_transformation = None

    # Initialize video capture (assuming Video class accepts an fps parameter)
    video = Video(input_path=video_path, output_fps=fps, output_path="new_vid.mp4")

    # List to store each frame with its detections
    results = []

    # Process each frame
    for i, frame in enumerate(video):
        # Compute noise level
        noise_level = compute_noise(frame)

        # Apply denoising
        if noise_level > 60:
            frame = apply_nlm_denoising(frame)
        # Object Detection
        ball_detections = ru.get_detections(
            yolo_detector, frame, class_id=0, confidence_threshold=0.3
        )
        player_detections = ru.get_detections(
            yolo_detector, frame, class_id=1, confidence_threshold=0.35
        )

        # Combine detections and update motion estimation
        detections = ball_detections + player_detections
        try:
            coord_transformation = ru.update_motion_estimator(
                motion_estimator=motion_estimator, detections=detections, frame=frame
            )
        except:
            pass

        # Tracking: update trackers for players and ball separately
        player_track_objects = player_tracker.update(
            detections=player_detections, coord_transformations=coord_transformation
        )
        ball_track_objects = ball_tracker.update(
            detections=ball_detections, coord_transformations=coord_transformation
        )

        # Convert tracked objects back to detection format
        player_detections = Converter.TrackedObjects_to_Detections_nor(
            player_track_objects, cls=1
        )
        ball_detections = Converter.TrackedObjects_to_Detections_nor(
            ball_track_objects, cls=0
        )

        # Append current frame and detections to results
        results.append((frame, ball_detections, player_detections))
        coord_transformations.append(coord_transformation)
        motion_estimators.append(motion_estimator)

    return results, motion_estimators, coord_transformations, video


if __name__ == "__main__":
    process_video("yolo8.pt", "jooooooooo.mp4", 30)
