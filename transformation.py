import cv2
import os
import torch
import yaml
import numpy as np
import json
from ultralytics import YOLO
import Transformation.inference as inf
import argparse
from sklearn.cluster import KMeans
import torchvision.models as models
from torchvision import transforms

# Field line coordinates remain global for simplicity.
LINES_COORDS = [
    [[0., 54.16, 0.], [16.5, 54.16, 0.]],
    [[16.5, 13.84, 0.], [16.5, 54.16, 0.]],
    [[16.5, 13.84, 0.], [0., 13.84, 0.]],
    [[88.5, 54.16, 0.], [105., 54.16, 0.]],
    [[88.5, 13.84, 0.], [88.5, 54.16, 0.]],
    [[88.5, 13.84, 0.], [105., 13.84, 0.]],
    [[0., 37.66, -2.44], [0., 30.34, -2.44]],
    [[0., 37.66, 0.], [0., 37.66, -2.44]],
    [[0., 30.34, 0.], [0., 30.34, -2.44]],
    [[105., 37.66, -2.44], [105., 30.34, -2.44]],
    [[105., 30.34, 0.], [105., 30.34, -2.44]],
    [[105., 37.66, 0.], [105., 37.66, -2.44]],
    [[52.5, 0., 0.], [52.5, 68, 0.]],
    [[0., 68., 0.], [105., 68., 0.]],
    [[0., 0., 0.], [0., 68., 0.]],
    [[105., 0., 0.], [105., 68., 0.]],
    [[0., 0., 0.], [105., 0., 0.]],
    [[0., 43.16, 0.], [5.5, 43.16, 0.]],
    [[5.5, 43.16, 0.], [5.5, 24.84, 0.]],
    [[5.5, 24.84, 0.], [0., 24.84, 0.]],
    [[99.5, 43.16, 0.], [105., 43.16, 0.]],
    [[99.5, 43.16, 0.], [99.5, 24.84, 0.]],
    [[99.5, 24.84, 0.], [105., 24.84, 0.]]
]

class FieldTransformer:
    """
    Handles the transformation from image coordinates to field (world) coordinates.
    """
    def __init__(self, H_inv: np.ndarray, field_offset=(105/2, 68/2)):
        self.H_inv = H_inv
        self.field_offset = field_offset

    def image_to_field_point(self, bbox):
        if bbox is None:
            return None
        # Compute the center of the bottom edge of the bounding box
        x_center = (bbox[0] + bbox[2]) / 2
        y_bottom = bbox[3]
        pt_img = np.array([x_center, y_bottom, 1]).reshape(3, 1)
        # Transform to field coordinates using the inverse homography
        pt_field = self.H_inv @ pt_img
        pt_field /= pt_field[2]  # Normalize homogeneous coordinates

        # Adjust with the field offset
        X = pt_field[0, 0] + self.field_offset[0]
        Y = pt_field[1, 0] + self.field_offset[1]
        return (X, Y)

    @staticmethod
    def compute_homography(P: np.ndarray):
        H = np.array([
            [P[0, 0], P[0, 1], P[0, 3]],
            [P[1, 0], P[1, 1], P[1, 3]],
            [P[2, 0], P[2, 1], P[2, 3]]
        ])
        return H

class CameraCalibrator:
    """
    Loads the camera calibration models and computes the projection matrix.
    """
    

    def __init__(self, cfg_path, cfg_line_path, kp_model_path, line_model_path):
        # Check if running in Kaggle environment
        if os.path.exists('/kaggle/working'):
            # Adjust paths for Kaggle environment
            cfg_path = '/kaggle/working/SoccerPossession/' + cfg_path
            cfg_line_path = '/kaggle/working/SoccerPossession/' + cfg_line_path
            kp_model_path = '/kaggle/working/SoccerPossession/' + kp_model_path
            line_model_path = '/kaggle/working/SoccerPossession/' + line_model_path

        # Load configuration and models
        self.cfg = yaml.safe_load(open(cfg_path, 'r'))
        self.cfg_l = yaml.safe_load(open(cfg_line_path, 'r'))

        self.model_kp = inf.get_cls_net(self.cfg)
        loaded_state = torch.load(kp_model_path, map_location=torch.device('cpu'))
        self.model_kp.load_state_dict(loaded_state)
        self.model_kp.eval()

        self.model_line = inf.get_cls_net_l(self.cfg_l)
        loaded_state_l = torch.load(line_model_path, map_location=torch.device('cpu'))
        self.model_line.load_state_dict(loaded_state_l)
        self.model_line.eval()

    def calibrate(self, frame, cam, kp_threshold, line_threshold):
        final_params_dict = inf.inference(cam, frame, self.model_kp, self.model_line, kp_threshold, line_threshold)
        P = inf.projection_from_cam_params(final_params_dict)
        return P

def process_field_transformation(precomputed_results, calibrator_cfgs):
    """
    precomputed_results: list of tuples
       Each tuple is (frame_image, <ignored>, detections)
       Each detection is a list:
         [class_id, x_min, y_min, x_max, y_max]
         where class_id is 0 for ball and 1 or 2 for player teams.
         
    calibrator_cfgs: dictionary containing the paths and thresholds for calibration.
    """
    # Use the first frame for calibration.
    first_frame = precomputed_results[0][0]
    print(f"type of frame{type(first_frame)}")
    frame_height, frame_width = first_frame.shape[:2]
    
    # Setup calibration.
    cam = inf.FramebyFrameCalib(iwidth=frame_width, iheight=frame_height, denormalize=True)
    calibrator = CameraCalibrator(calibrator_cfgs["cfg_path"],
                                  calibrator_cfgs["cfg_line_path"],
                                  calibrator_cfgs["kp_model_path"],
                                  calibrator_cfgs["line_model_path"])
    kp_threshold = calibrator_cfgs["kp_threshold"]
    line_threshold = calibrator_cfgs["line_threshold"]

    # Calibrate using the first frame.
    P = calibrator.calibrate(first_frame, cam, kp_threshold, line_threshold)
    H = FieldTransformer.compute_homography(P)
    H_inv = np.linalg.inv(H)
    transformer = FieldTransformer(H_inv)

    output_results = []
    # Process each frame.
    for frame_idx, (frame, _, detections) in enumerate(precomputed_results):
        players_with_field = []
        ball_field_position = None

        for detection in detections:
            # detection: [class_id, x_min, y_min, x_max, y_max]
            class_id = int(detection[0])
            bbox = [float(coord) for coord in detection[1:]]  # [x_min, y_min, x_max, y_max]
            if class_id in [1, 2]:  # players (already classified)
                field_pos = transformer.image_to_field_point(bbox)
                if field_pos:
                    players_with_field.append({
                        "class_id": class_id,
                        "field_position": list(field_pos)
                    })
            elif class_id == 0:  # ball detection
                field_pos = transformer.image_to_field_point(bbox)
                if field_pos:
                    ball_field_position = field_pos

        frame_result = {
            "frame_index": frame_idx,
            "players": players_with_field,
            "ball": {"class_id": 0, "field_position": list(ball_field_position)}
                    if ball_field_position is not None else None
        }
        output_results.append(frame_result)
    return output_results

def main():
    # Example: load your precomputed results from another module or file.
    # Here we assume that `precomputed_results` is defined externally.
    # For demonstration, we will assume it is loaded from a JSON file.
    # (Make sure that the JSON file correctly reflects the structure expected.)
    precomputed_results_path = r"precomputed_results.json"
    try:
        with open(precomputed_results_path, "r") as f:
            precomputed_results = json.load(f)
    except Exception as e:
        print("Error loading precomputed results:", e)
        return

    # The calibration configuration paths and thresholds.
    calibrator_cfgs = {
        "cfg_path": r"Transformation/config/hrnetv2_w48.yaml",
        "cfg_line_path": r"Transformation/config/hrnetv2_w48_l.yaml",
        "kp_model_path": r"resources/SV_FT_TSWC_kp",
        "line_model_path": r"resources/SV_FT_TSWC_lines",
        "kp_threshold": 0.1486,
        "line_threshold": 0.3880
    }

    # Process the field transformation using the precomputed detections.
    results = process_field_transformation(precomputed_results, calibrator_cfgs)

    # Save the transformed results to a JSON file.
    json_output_path = r"results_field_transformed.json"
    with open(json_output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {json_output_path}")

if __name__ == '__main__':
    main()
