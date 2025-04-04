import json
import time
from tracker import process_video
from cluster_time_improve import main_multi_frame
from transformation import process_field_transformation
from possessionCalculation import CalculatePossession
import warnings
from videoDraw import draw_bounding_boxes_on_frames , save_video_from_frames
warnings.filterwarnings("ignore")

def measure_time(func, *args, process_name="Process"):
    """Helper function to measure execution time."""
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{process_name} completed in {elapsed_time:.4f} seconds")
    return result

# TRACKING
tracking_results, coord_transformations, nor_ball_detections = measure_time(process_video, r"resources\yolo8.pt", r"resources\manc3.mp4", 20, process_name="Tracking")

# CLUSTERING
results_with_class_ids = measure_time(main_multi_frame, tracking_results, process_name="Clustering")

# Calibration configuration
calibrator_cfgs = {
    "cfg_path": r"Transformation/config/hrnetv2_w48.yaml",
    "cfg_line_path": r"Transformation/config/hrnetv2_w48_l.yaml",
    "kp_model_path": r"resources/SV_FT_TSWC_kp",
    "line_model_path": r"resources/SV_FT_TSWC_lines",
    "kp_threshold": 0.1486,
    "line_threshold": 0.3880
}

# FIELD TRANSFORMATION
results = measure_time(process_field_transformation, results_with_class_ids, calibrator_cfgs, process_name="Field Transformation")

# POSSESSION CALCULATION
yardTL, yardTR, yardBL, yardBR = [29.0, 17.0], [45.5, 17.0], [29.0, 26.0], [45.5, 26.0]
poss, team_possession = measure_time(CalculatePossession, results, yardTL, yardTR, yardBL, yardBR, process_name="Possession Calculation")

print("Possession Results:", poss)
visualize = draw_bounding_boxes_on_frames(nor_ball_detections, results_with_class_ids, coord_transformations, team_possession)


save_video_from_frames(visualize, output_path="detection_results.mp4")

# Optional: Save results to a JSON file
# json_output_path = r"results_field_transformed.json"
# with open(json_output_path, "w") as f:
#     json.dump(results, f, indent=4)
# print(f"Results saved to {json_output_path}")
