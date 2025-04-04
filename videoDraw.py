import cv2
import numpy as np
import random
import PIL
from PIL import Image
from tracking.soccer.draw import AbsolutePath
from tracking.inference.converter import Converter
from norfair import Detection, Tracker
from run_utils import get_main_ball
path = AbsolutePath()

def yolo_to_norfair(yolo_dets):
    norfair_detections = []
    
    for det in yolo_dets:
        center_x, center_y, width, height, confidence = det
        
        # Convert YOLO box center to Norfair points
        # Norfair expects points as (n_points, 2) array
        # We'll use the center point of the bounding box
        points = np.array([[center_x, center_y]])
        
        # Confidence score as array matching points length
        scores = np.array([confidence])
        
        # Create Norfair Detection object
        detection = Detection(
            points=points,
            scores=scores,
            # Optional: add any extra data you want to track
            data={"width": width, "height": height}
            # You could add label="person" or similar if your YOLO provides class info
        )
        
        norfair_detections.append(detection)
    
    return norfair_detections

def draw_bounding_boxes_on_frames(results, results_with_class_ids, coord_transformations, team_possession):
    """Draw bounding boxes and ball path on frames using OpenCV and Norfair's path.draw."""
    
    class_id_colors = {}  # Store unique colors for each class ID
    ball_color = (0, 255, 255)  # Yellow for ball

    for i, (frame, ball_detections, updated_boxes) in enumerate(results_with_class_ids):
        print(f"frame number {i}")
        print(f"results  {results[i]}")
        
        # Ensure frame is in OpenCV format (NumPy array)
        if not isinstance(frame, np.ndarray):
            print(f"Frame {i} is not a NumPy array!")
            continue

        # Create a copy to preserve original frame
        frame_copy = frame.copy()

        # Draw bounding boxes for players
        for box in updated_boxes:
            class_id, x1, y1, x2, y2 = box
            if class_id not in class_id_colors:
                class_id_colors[class_id] = tuple(random.randint(50, 255) for _ in range(3))

            color = class_id_colors[class_id]
            cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Handle ball detections
#         ball_detections = results[i][1]
#         ball_detections = yolo_to_norfair(ball_detections)
        ball = get_main_ball(results[i])

        # Draw ball path and bounding box
        if ball and ball.detection:
            # Convert to PIL for path drawing
            frame_pil = PIL.Image.fromarray(frame_copy)
            frame_pil = path.draw(
                img=frame_pil,
                detection=ball.detection,
                coord_transformations=coord_transformations,
            )
            # Convert back to OpenCV format
            frame_copy = np.array(frame_pil)
            
        # Update the frame in results_with_class_ids
        results_with_class_ids[i] = (frame_copy, ball_detections, updated_boxes)

    return results_with_class_ids

def save_video_from_frames(results_with_class_ids, output_path="output.mp4", fps=30):
    """Creates and saves a video from processed frames."""
    
    first_frame = results_with_class_ids[0][0]  # First frame as NumPy array
    height, width, _ = first_frame.shape

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame, _, _ in results_with_class_ids:
        out.write(frame)  # Write frame to video

    out.release()
    print(f"Video saved as {output_path}")

# Process frames & save video
# results_with_class_ids = draw_bounding_boxes_on_frames(results_with_class_ids)
# save_video_from_frames(results_with_class_ids, output_path="detection_results.mp4", fps=30)
