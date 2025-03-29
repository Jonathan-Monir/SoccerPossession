import cv2
import numpy as np
import os
import re
from collections import Counter
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

# -------------------------
# Feature Extraction
# -------------------------
def extract_deep_features(image, model):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features.flatten()

# -------------------------
# Cropping and Parsing
# -------------------------
def crop_players(image, boxes, image_shape):
    player_images = []
    img_h, img_w = image_shape[:2]
    for box in boxes:
        label, x_center, y_center, width, height = box
        if label == 1:  # Assuming label '1' corresponds to players
            x = int((x_center - width / 2) * img_w)
            y = int((y_center - height / 2) * img_h)
            w = int(width * img_w)
            h = int(height * img_h)
            player_crop = image[max(0, y):min(y + h, img_h), max(0, x):min(x + w, img_w)]
            player_images.append(player_crop)
    return player_images

def parse_yolo_labels(file_path):
    bounding_boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = list(map(float, line.split()))
            cls, x1, x2, y1, y2 = parts
            bounding_boxes.append([cls, x1, y1, x2, y2])
    return bounding_boxes

# -------------------------
# Color Extraction Functions
# -------------------------
def calculate_color_distance(color1, color2):
    hsv1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_BGR2HSV)[0][0]
    hsv2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_BGR2HSV)[0][0]
    rgb_dist = np.sqrt(np.sum((color1 - color2) ** 2))
    h_dist = min(abs(hsv1[0] - hsv2[0]), 180 - abs(hsv1[0] - hsv2[0])) / 90.0
    s_dist = abs(hsv1[1] - hsv2[1]) / 255.0
    v_dist = abs(hsv1[2] - hsv2[2]) / 255.0
    hsv_dist = np.sqrt(4 * h_dist ** 2 + s_dist ** 2 + v_dist ** 2)
    return 0.5 * rgb_dist + 0.5 * hsv_dist * 255

def apply_masking(crop):
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv_crop, lower_green, upper_green)
    field_sample = hsv_crop[hsv_crop.shape[0] // 2:, :, :]
    field_hue = np.median(field_sample[:, :, 0])
    field_lower = np.array([max(0, field_hue - 10), 30, 30])
    field_upper = np.array([min(180, field_hue + 10), 255, 255])
    field_mask = cv2.inRange(hsv_crop, field_lower, field_upper)
    ground_mask = cv2.bitwise_or(green_mask, field_mask)
    player_mask = cv2.bitwise_not(ground_mask)
    lower_uniform = np.array([0, 30, 40])
    upper_uniform = np.array([180, 255, 255])
    color_mask = cv2.inRange(hsv_crop, lower_uniform, upper_uniform)
    final_mask = cv2.bitwise_and(player_mask, color_mask)
    kernel = np.ones((3, 3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    masked_crop = cv2.bitwise_and(crop, crop, mask=final_mask)
    pixels = masked_crop.reshape((-1, 3))
    pixels = pixels[np.any(pixels != 0, axis=1)]
    return pixels




def extract_dominant_colors(crop, n_colors=3, debug=False):
    pixels = apply_masking(crop)
    if debug:
        print(f"Number of pixels after masking: {len(pixels)}")
    
    # Fallback if we don't have enough pixels
    if len(pixels) < 10:
        h, w = crop.shape[:2]
        center_crop = crop[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        pixels = center_crop.reshape(-1, 3)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=22, n_init=100, max_iter=500)
    kmeans.fit(pixels)

    labels = kmeans.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    centers = kmeans.cluster_centers_

    # Debug: Show cluster info
    if debug:
        print("\nClustered Colors (BGR) and their counts:")
        for center, count in zip(centers, counts):
            print(f"  Color: {tuple(map(int, center))}, Count: {count}")
        visualize_colors(centers, counts)

    # Sort colors by frequency
    color_counts = list(zip(centers, counts))
    color_counts.sort(key=lambda x: x[1], reverse=True)

    dominant_colors = []
    for color, count in color_counts:
        color_tuple = tuple(map(int, color))
        if debug:
            print(f"Selected color: {color_tuple} with count {count}")
        dominant_colors.append(color_tuple)
        if len(dominant_colors) == 2:
            break

    # Ensure two colors are returned
    while len(dominant_colors) < 2:
        dominant_colors.append((0, 0, 0))

    if debug:
        visualize_extraction(crop, dominant_colors)

    return dominant_colors

def visualize_colors(centers, counts):
    """Display clustered colors as a horizontal bar chart."""
    sorted_colors = [tuple(map(int, color)) for color, _ in sorted(zip(centers, counts), key=lambda x: x[1], reverse=True)]
    fig, ax = plt.subplots(figsize=(8, 2))
    
    color_bar = np.zeros((50, len(sorted_colors) * 50, 3), dtype=np.uint8)
    for i, color in enumerate(sorted_colors):
        color_bar[:, i * 50:(i + 1) * 50] = color

    plt.imshow(cv2.cvtColor(color_bar, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Clustered Colors")
    plt.show()


def visualize_extraction(crop, filtered_colors):
    """
    Displays the cropped image, the masked image, and a color bar of the final extracted colors.
    """
    # Convert crop from BGR to RGB for proper display in matplotlib
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    # Apply masking and obtain the masked image
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv_crop, lower_green, upper_green)
    masked_crop = cv2.bitwise_and(crop, crop, mask=cv2.bitwise_not(mask))
    masked_crop_rgb = cv2.cvtColor(masked_crop, cv2.COLOR_BGR2RGB)

    # Create a color bar for the filtered colors
    n_colors = len(filtered_colors)
    color_bar = np.zeros((50, n_colors * 50, 3), dtype=np.uint8)
    for i, color in enumerate(filtered_colors):
        color_bar[:, i * 50:(i + 1) * 50] = color
    color_bar_rgb = cv2.cvtColor(color_bar, cv2.COLOR_BGR2RGB)

    # Create a subplot with three panels: original image, masked image, and color bar
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(crop_rgb)
    axes[0].axis("off")
    axes[0].set_title("Original Cropped Image")

    axes[1].imshow(masked_crop_rgb)
    axes[1].axis("off")
    axes[1].set_title("Masked Image")

    axes[2].imshow(color_bar_rgb)
    axes[2].axis("off")
    axes[2].set_title("Extracted Colors")

    plt.tight_layout()
    plt.show()

def visualize_colors(centers, counts):
    """Display clustered colors as a bar chart."""
    sorted_colors = [tuple(map(int, color)) for color, _ in sorted(zip(centers, counts), key=lambda x: x[1], reverse=True)]
    fig, ax = plt.subplots(figsize=(8, 2))
    
    color_bar = np.zeros((50, len(sorted_colors) * 50, 3), dtype=np.uint8)
    for i, color in enumerate(sorted_colors):
        color_bar[:, i * 50:(i + 1) * 50] = color

    plt.imshow(cv2.cvtColor(color_bar, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def extract_player_colors(image, bounding_boxes):
    player_colors = []
    other_boxes = []
    if image is None:
        print("Error: Could not load image.")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for box in bounding_boxes:
        # This part is kept exactly as provided:
        cls, x1, x2, y1, y2 = box
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        
        # Crop and display the image
        # Uncomment the following lines to visualize each cropped region
        if cls in range(1, 6):  # Process only player boxes
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue
            dominant_colors = extract_dominant_colors(crop)
            player_colors.append((dominant_colors, box))
        else:
            other_boxes.append(box)
    return player_colors, other_boxes

# -------------------------
# Clustering and Classification
# -------------------------
def classify_players_by_features(player_features, n_teams=2):
    kmeans = KMeans(n_clusters=n_teams, random_state=42, n_init=10)
    labels = kmeans.fit_predict(player_features)
    return labels

# -------------------------
# Visualization
# -------------------------
def visualize_results(vis_image, player_colors, team_labels, other_boxes, frame_index):
    COLORS = {
        0: (255, 255, 255),  # Ball - Gray
        1: (0, 0, 255),      # T1 - Red
        2: (255, 0, 0),      # T2 - Blue
        3: (0, 255, 0),      # REF - Green
        4: (255, 255, 0),    # GK1 - Cyan
        5: (0, 255, 255)     # GK2 - Yellow
    }
    LABELS = {
        0: "Ball",
        1: "T1",
        2: "T2",
        3: "REF",
        4: "GK1",
        5: "GK2"
    }
    
    cv2.putText(vis_image, f'Frame: {frame_index}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # This part is kept exactly as provided:
    for (color, box), team_label in zip(player_colors, team_labels):
        cls, x_center, y_center, width, height = box
        x1 = int(x_center)
        y1 = int(y_center)
        x2 = int(width)
        y2 = int(height)
        label_idx = int(team_label if team_label > 2 else team_label + 1)
        box_color = COLORS[label_idx]
        label_text = LABELS[label_idx]
        cv2.rectangle(vis_image, (x1, x2), (y1, y2), box_color, 2)
        cv2.putText(vis_image, label_text, (x1, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    for box in other_boxes:
        cls, x1, x2, y1, y2 = box
        cls_idx = int(cls)
        box_color = COLORS.get(cls_idx, (128, 128, 128))
        label_text = LABELS.get(cls_idx, f"C{cls_idx}")
        cv2.rectangle(vis_image, (int(x1), int(x2)), (int(y1), int(y2)), box_color, 2)
        cv2.putText(vis_image, label_text, (int(x1), int(y2) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(vis_image_rgb)
    plt.axis('off')
    plt.title('Team Classification with Deep Features')
    plt.show()

# -------------------------
# Multi-Frame Clustering (No File Saving)
# -------------------------
def multi_frame_cluster(results, model=None):
    """
    Process multiple frames to produce consistent team classification across frames.
    If a deep model is provided, deep features will be extracted from each player crop.
    Otherwise, color features will be used.
    Args:
        results: A list of tuples (frame, ball_detections, player_detections)
                 where player_detections are in YOLO format [cls, x1, y1, x2, y2].
        model: (Optional) Pre-trained model for deep feature extraction.
    Returns:
        final_results: A list of tuples (frame, ball_detections, updated_player_detections)
                       with updated class IDs.
    """
    all_features = []   # List to hold features for clustering
    features_info = []  # List to hold (frame_index, original_box) for mapping back

    for frame_idx, (frame, ball_detections, player_detections) in enumerate(results):
        if model is not None:
            for box in player_detections:
                # box is assumed as [cls, x1, y1, x2, y2]
                _, x1, x2, y1, y2 = box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                feat = extract_deep_features(crop, model)
                all_features.append(feat)
                features_info.append((frame_idx, box))
        else:
            player_colors, _ = extract_player_colors(frame, player_detections)
            print("Hi")
            for colors, box in player_colors:
                feat = np.array(colors).flatten()
                all_features.append(feat)
                features_info.append((frame_idx, box))

    all_features = np.array(all_features)
    team_labels_global = classify_players_by_features(all_features, n_teams=2)

    new_player_detections = {i: [] for i in range(len(results))}
    for global_idx, label in enumerate(team_labels_global):
        frame_idx, original_box = features_info[global_idx]
        new_label = label + 1  # Map label to team IDs (0 reserved for ball)
        new_box = [new_label] + original_box[1:]
        new_player_detections[frame_idx].append(new_box)

    final_results = []
    for idx, (frame, ball_detections, _) in enumerate(results):
        updated_boxes = new_player_detections.get(idx, []) + ball_detections
        updated_boxes.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
        final_results.append((frame, ball_detections, updated_boxes))
    return final_results

# -------------------------
# Example Main Function Returning Results
# -------------------------
def main_multi_frame():
    results = []
    frames_dir = 'frames'   # Update this path if needed
    labels_dir = 'labels'   # Update this path if needed

    frame_files = sorted(os.listdir(frames_dir), key=lambda f: int(re.findall(r'\d+', f)[0]))
    
    for frame_file in frame_files[7:15]:
        frame_path = os.path.join(frames_dir, frame_file)
        label_path = os.path.join(labels_dir, frame_file.replace('.jpg', '.txt'))
        
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        
        bounding_boxes = parse_yolo_labels(label_path)
        ball_detections = [box for box in bounding_boxes if box[0] == 0]
        player_detections = [box for box in bounding_boxes if box[0] == 1]
        results.append((frame, ball_detections, player_detections))
    
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    model = Model(inputs=base_model.input, outputs=base_model.output)
    
    updated_results = multi_frame_cluster(results, model=None)
    
    for frame_idx, (frame, ball_detections, updated_boxes) in enumerate(updated_results):
        player_colors, other_boxes = extract_player_colors(frame, updated_boxes)
        team_labels = []
        for box in updated_boxes:
            if box[0] in [1, 2, 3, 4, 5]:
                team_labels.append(box[0] if box[0] > 2 else box[0] - 1)
        visualize_results(frame.copy(), player_colors, team_labels, other_boxes, frame_idx)
    
    return updated_results

if __name__ == '__main__':
    results_with_class_ids = main_multi_frame()
    print("Multi-frame clustering complete. Processed results returned.")
