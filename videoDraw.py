import cv2
import numpy as np
import random

def draw_bounding_boxes_on_frames(results_with_class_ids):
    """Draw bounding boxes on frames using OpenCV."""
    
    class_id_colors = {}  # Store unique colors for each class ID
    ball_color = (0, 255, 255)  # Yellow for ball

    for i, (frame, ball_detections, updated_boxes) in enumerate(results_with_class_ids):
        # Ensure frame is in OpenCV format (NumPy array)
        if not isinstance(frame, np.ndarray):
            print(f"Frame {i} is not a NumPy array!")
            continue

        # Draw bounding boxes for players
        for box in updated_boxes:
            class_id, x1, y1, x2, y2 = box  # Correct unpacking
            if class_id not in class_id_colors:
                class_id_colors[class_id] = tuple(random.randint(50, 255) for _ in range(3))

            color = class_id_colors[class_id]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # Draw rectangle

        # Draw bounding boxes for the ball
        for ball_box in ball_detections:
            ball_class_id, x1, y1, x2, y2 = ball_box  # Corrected unpacking
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), ball_color, 2)

    return results_with_class_ids  # Optionally return if you want to save/show later

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
