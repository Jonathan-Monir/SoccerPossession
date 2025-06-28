import cv2
import numpy as np

def vid(results_with_class_ids, team1_color, team2_color, output_path="output_video.mp4", fps=30):
    """
    Draw bounding boxes on video frames for player and ball detections and save as .mp4.
    
    Args:
        results_with_class_ids: List of tuples (frame, ball_detections, player_detections)
        team1_color: BGR tuple for class ID 1 (team 1)
        team2_color: BGR tuple for class ID 2 (team 2)
        output_path: Path to save the output video (.mp4)
        fps: Frames per second for the video
    
    Returns:
        Path to saved video
    """
    frames = []
    # Frame size from first frame
    height, width, _ = results_with_class_ids[0][0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_idx, (frame, ball_detections, player_detections) in enumerate(results_with_class_ids):
        # Draw player detections (List of lists: [class_id, x1, y1, x2, y2])
        for det in player_detections:
            class_id, x1, y1, x2, y2 = det
            if class_id ==0 or class_id==3:
                print(f"at frame idx: {frame_idx}, ball det is at: {x1,x2,y1,y2}, class_id: {class_id}")
            color = team1_color if class_id == 1 else team2_color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


        # Write frame to output
        out.write(frame)
        frames.append(frame)

    out.release()
    return output_path,frames
