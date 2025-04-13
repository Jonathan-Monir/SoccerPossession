import json
import time
from typing import List
from PIL import Image
import norfair
from norfair.camera_motion import MotionEstimator
from Transformation.utils.utils_heatmap import coords_to_dict
from tracker import process_video
from cluster_time_improve import main_multi_frame
from transformation import process_field_transformation
from possessionCalculation import CalculatePossession
import warnings
from videoDrawImprove import draw_bounding_boxes_on_frames, save_video_from_frames
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
results_tracking, motion_estimators, coord_transformations, video = measure_time(process_video, r"resources\yolo8.pt", r"resources\manc3.mp4", 20, process_name="Tracking")


team1 = 0
team2 = 0

team_poss_list = [1]
motion_estimators = 1
# visualize = draw_bounding_boxes_on_frames(results_with_class_ids, team1_color, team2_color, team_poss_list)
visualize = draw_bounding_boxes_on_frames(results_tracking, team1, team2, team_poss_list, motion_estimators, coord_transformations, video)

# CLUSTERING
results_with_class_ids, team1_color, team2_color = measure_time(main_multi_frame, results_tracking, process_name="Clustering")

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

visualize = draw_bounding_boxes_on_frames(results_tracking, (0,244,244), (0,123,123), [], motion_estimators, coord_transformations, video)

# POSSESSION CALCULATION
yardTL, yardTR, yardBL, yardBR = [29.0, 17.0], [45.5, 17.0], [29.0, 26.0], [45.5, 26.0]

poss, team_poss_list = measure_time(CalculatePossession, results, yardTL, yardTR, yardBL, yardBR, process_name="Possession Calculation")

print("Possession Results:", poss)



# visualize = draw_bounding_boxes_on_frames(results_with_class_ids, team1_color, team2_color, team_poss_list)
visualize = draw_bounding_boxes_on_frames(results_tracking, team1_color, team2_color, team_poss_list, motion_estimator, coord_transformations, video)



# Optional: Save results to a JSON file
# json_output_path = r"results_field_transformed.json"
# with open(json_output_path, "w") as f:
#     json.dump(results, f, indent=4)
# print(f"Results saved to {json_output_path}")
