import torch
import cv2
from ultralytics import YOLO
# Replace the following imports with your actual utility modules
# e.g., if your project has ru.py, tracker.py, motion.py, converter.py, video.py, import them accordingly
# Example:
# import ru
# from tracker import Tracker
# from motion_estimator import MotionEstimator
# from converter import Converter
# from video import Video
# from utils import mean_euclidean

# For now, assuming all utilities are in modules named appropriately:
import ru
from tracker import Tracker
from motion_estimator import MotionEstimator
from converter import Converter
from video import Video
from utils import mean_euclidean

def process_video(yolo_path, video_path, target_fps, last_frame, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coord_transformations, motion_estimators = [], []

    # Initialize YOLO detector
    yolo_detector = YOLO(yolo_path)
    yolo_detector.model.to(device)

    # Initialize trackers and motion estimator
    player_tracker = Tracker(distance_function=mean_euclidean,
                             distance_threshold=250,
                             initialization_delay=3,
                             hit_counter_max=90)
    ball_tracker = Tracker(distance_function=mean_euclidean,
                           distance_threshold=150,
                           initialization_delay=20,
                           hit_counter_max=2000)
    motion_estimator = MotionEstimator()

    # Open video to get FPS and frame count
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if last_frame == -1:
        last_frame = total_frames - 1
    if target_fps <= 0 or target_fps > orig_fps:
        raise ValueError(f"target_fps must be >0 and <= source FPS ({orig_fps})")

    skip_interval = int(round(orig_fps / target_fps))
    video = Video(input_path=video_path, output_path="new_vid.mp4")

    results = []
    batch_frames = []
    frame_indices = []

    # Collect frames, then process in batches
    for i, frame in enumerate(video):
        if i > last_frame or i >= total_frames:
            break
        if skip_interval > 1 and (i % skip_interval) != 0:
            continue

        batch_frames.append(frame)
        frame_indices.append(i)

        # When batch is full or at end, run detection
        if len(batch_frames) == batch_size or i == last_frame:
            # Batch inference
            batch_results = yolo_detector(batch_frames)

            for idx_in_batch, result in enumerate(batch_results):
                frame_idx = frame_indices[idx_in_batch]
                frame = batch_frames[idx_in_batch]

                # Use .boxes.data: [x1, y1, x2, y2, conf, cls]
                data = result.boxes.data.cpu().numpy()
                # Separate detections by class with thresholds
                ball_detections = [d[:5].tolist() for d in data if int(d[5]) == 0 and d[4] >= 0.3]
                player_detections = [d[:5].tolist() for d in data if int(d[5]) == 1 and d[4] >= 0.35]

                # Motion estimation update
                detections = ball_detections + player_detections
                try:
                    coord_trans = ru.update_motion_estimator(
                        motion_estimator=motion_estimator,
                        detections=detections,
                        frame=frame
                    )
                except Exception:
                    coord_trans = None

                # Tracking
                player_tracks = Converter.TrackedObjects_to_Detections_nor(
                    player_tracker.update(detections=player_detections, coord_transformations=coord_trans), cls=1)
                ball_tracks = Converter.TrackedObjects_to_Detections_nor(
                    ball_tracker.update(detections=ball_detections, coord_transformations=coord_trans), cls=0)
                if not ball_tracks:
                    ball_tracks = ball_detections

                # Save results
                results.append((frame, ball_tracks, player_tracks))
                coord_transformations.append(coord_trans)
                motion_estimators.append(motion_estimator)

            # Reset batch
            batch_frames.clear()
            frame_indices.clear()

    return results, motion_estimators, coord_transformations, video

if __name__ == "__main__":
    process_video("yolo8.pt", "jooooooooo.mp4", 30)
