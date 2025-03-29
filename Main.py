import json
from tracker import process_video
from cluster_time_improve import main_multi_frame
from transformation import process_field_transformation

# TRACKING
results = process_video(r"resources\yolo8.pt", r"resources\manc2.mp4", 20)
print("Tracking completed")

# CLUSTER
results_with_class_ids = main_multi_frame(results)
print("Clustering completed")

# The calibration configuration paths and thresholds.
calibrator_cfgs = {
    "cfg_path": r"Transformation/config/hrnetv2_w48.yaml",
    "cfg_line_path": r"Transformation/config/hrnetv2_w48_l.yaml",
    "kp_model_path": r"resources/SV_FT_TSWC_kp",
    "line_model_path": r"resources/SV_FT_TSWC_lines",
    "kp_threshold": 0.1486,
    "line_threshold": 0.3880
}

# FEILD TRANSFORMATION
results = process_field_transformation(results_with_class_ids, calibrator_cfgs)





json_output_path = r"results_field_transformed.json"
with open(json_output_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"Results saved to {json_output_path}")
