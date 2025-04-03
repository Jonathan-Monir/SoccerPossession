import os
import cv2
import numpy as np
import PIL
from ultralytics import YOLO
from norfair import Tracker
from norfair.camera_motion import MotionEstimator
from norfair.distances import mean_euclidean

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

def delete_folder_contents(folder: str) -> None:
    """
    Deletes all files and subdirectories in the specified folder.

    Parameters:
        folder (str): The path to the folder whose contents will be deleted.
    """
    # Check if the folder exists
    if not os.path.exists(folder):
        print(f"The folder '{folder}' does not exist.")
        return

    # Iterate through each item in the folder
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        try:
            # Remove file or symbolic link
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            # Remove directory and its contents
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Failed to delete '{item_path}'. Reason: {e}")

"""## Video extraction"""
def extract_frames(video_path, output_dir, fps, start_second=0, end_second=-1):
    """Extracts frames from a video file within a specified time range and saves them to an output directory.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Path to the output directory.
        fps (int): Desired frames per second.
        start_second (int): Starting second to extract frames from.
        end_second (int): Ending second to extract frames until (-1 for full video).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    delete_folder_contents(output_dir)
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    if end_second == -1 or end_second > duration:
        end_second = duration

    start_frame = int(start_second * video_fps)
    end_frame = int(end_second * video_fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_count = start_frame
    while frame_count < end_frame:
        ret, frame = cap.read()

        if not ret:
            break

        if (frame_count - start_frame) % int(video_fps / fps) == 0:
            cv2.imwrite(os.path.join(output_dir, f'frame_{frame_count}.jpg'), frame)

        frame_count += 1

    cap.release()

# extract_frames(video_path, output_dir, fps)

def process_video(yolo_path, video_path, fps):
    # Initialize YOLO detector with the given model path
    yolo_detector = YOLO(yolo_path)
    
    # Initialize trackers and motion estimator
    player_tracker = Tracker(distance_function=mean_euclidean, distance_threshold=250)
    ball_tracker = Tracker(distance_function=mean_euclidean, distance_threshold=150)
    motion_estimator = MotionEstimator()
    coord_transformations = None

    # Initialize video capture (assuming Video class accepts an fps parameter)
    video = Video(input_path=video_path, output_fps=fps)

    # List to store each frame with its detections
    results = []

    # Process each frame
    for i, frame in enumerate(video):
        print(type(frame))
        # Object Detection
        ball_detections = ru.get_detections(
            yolo_detector, frame, class_id=0, confidence_threshold=0.3
        )
        player_detections = ru.get_detections(
            yolo_detector, frame, class_id=1, confidence_threshold=0.35
        )

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

        # Convert tracked objects back to detection format
        player_detections = Converter.TrackedObjects_to_Detections(player_track_objects,cls=1)
        ball_detections = Converter.TrackedObjects_to_Detections(ball_track_objects,cls=0)

        # Append current frame and detections to results
        results.append((frame, ball_detections, player_detections))

    return results

if __name__ == "__main__":
    process_video("yolo8.pt","manc.mp4",30)
