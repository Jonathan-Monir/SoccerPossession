import numpy as np
import os
import cv2

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

def generate_field_visualization(results_with_transform, temp_video_path, fps, unique_id, final_team1_color, final_team2_color):
    """
    Generate field transformation visualization video from processed results.
    Added team colors as parameters.
    """
    print("Starting field transformation visualization...")
    print(f"Input data type: {type(results_with_transform)}")
    print(f"Number of results: {len(results_with_transform)}")
    
    # Debug first few results
    for i, result in enumerate(results_with_transform[:2]):
        print(f"Result {i} keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        if isinstance(result, dict) and "players" in result:
            print(f"Result {i} players count: {len(result['players'])}")
            if result['players']:
                print(f"First player data: {result['players'][0]}")
    
    # Create output path
    field_filename = f"field_{unique_id}.mp4"
    field_output_path = os.path.join(FIELD_VIDEOS_DIR, field_filename)
    
    # Parameters for visualization
    scale = 10
    ss_factor = 4
    margin_left = margin_right = margin_top = margin_bottom = 2
    
    # Open original video to get frame info
    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {temp_video_path}")
    
    # Get total frame count from original video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Original video has {total_frames} frames")
    
    # Calculate correct output dimensions
    field_width, field_height = 105, 68
    output_width = int((field_width + margin_left + margin_right) * scale)
    output_height = int((field_height + margin_top + margin_bottom) * scale)
    
    print(f"Output video dimensions: {output_width} x {output_height}")
    
    # Create video writer for field visualization
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(field_output_path, fourcc, fps, (output_width, output_height))
    
    # Debug the input data structure
    print(f"Processing {len(results_with_transform)} transformation results...")
    print("Sample of first few results:")
    for i, result in enumerate(results_with_transform[:3]):
        print(f"Result {i}: {result}")
    
    # Group results by frame with better error handling
    frame_data = {}
    max_frame_num = 0
    total_players_found = 0
    total_balls_found = 0
    
    # Add counters for each team
    team1_players_count = 0
    team2_players_count = 0
    
    # Process each result
    for result_idx, frame_result in enumerate(results_with_transform):
        try:
            # Handle different possible data structures
            frame_num = frame_result.get("frame_index", frame_result.get("frame", result_idx))
            max_frame_num = max(max_frame_num, frame_num)
            
            if frame_num not in frame_data:
                frame_data[frame_num] = {"players": [], "ball": None}
            
            # Extract players - handle multiple possible structures
            players = []
            if "players" in frame_result:
                players = frame_result["players"]
            elif "detections" in frame_result:
                # If detections contain both players and ball
                detections = frame_result["detections"]
                players = [d for d in detections if d.get("class") == "player" or d.get("type") == "player"]
            
            # Process players
            for player in players:
                if isinstance(player, dict):
                    # Try different field position key names
                    field_pos = None
                    team = None
                    
                    for pos_key in ["field_position", "field_coordinates", "world_position", "position"]:
                        if pos_key in player:
                            field_pos = player[pos_key]
                            break
                    
                    # Use class_id directly from the clustering results
                    if "class_id" in player:
                        team = player["class_id"] - 1  # Convert class_id (1,2) to team (0,1)
                        print(f"Found player with class_id {player['class_id']}, assigned to team {team}")
                    else:
                        # Fallback to other team keys
                        for team_key in ["team", "team_id", "cluster"]:
                            if team_key in player:
                                team = player[team_key]
                                break
                    
                    if field_pos and team is not None:
                        # Ensure field_pos is a list/tuple with at least 2 elements
                        if isinstance(field_pos, (list, tuple)) and len(field_pos) >= 2:
                            X, Y = float(field_pos[0]), float(field_pos[1])
                            
                            # Validate coordinates are within reasonable bounds
                            if 0 <= X <= field_width and 0 <= Y <= field_height:
                                frame_data[frame_num]["players"].append({
                                    "field_position": (X, Y),
                                    "team": int(team)
                                })
                                total_players_found += 1
                                
                                # Count players per team
                                if int(team) == 0:
                                    team1_players_count += 1
                                elif int(team) == 1:
                                    team2_players_count += 1
                            else:
                                print(f"Player out of field bounds: ({X:.1f}, {Y:.1f}) in frame {frame_num}")
                        else:
                            print(f"Invalid field position format: {field_pos}")
                    else:
                        print(f"Missing field position or team info: field_pos={field_pos}, team={team}")
                        print(f"Player data: {player}")
            
            # Extract ball - handle multiple possible structures
            ball_pos = None
            if "ball" in frame_result and frame_result["ball"]:
                ball_data = frame_result["ball"]
                if isinstance(ball_data, dict):
                    for pos_key in ["field_position", "field_coordinates", "world_position", "position"]:
                        if pos_key in ball_data:
                            ball_pos = ball_data[pos_key]
                            break
                elif isinstance(ball_data, (list, tuple)):
                    ball_pos = ball_data
            
            if ball_pos and isinstance(ball_pos, (list, tuple)) and len(ball_pos) >= 2:
                X, Y = float(ball_pos[0]), float(ball_pos[1])
                
                # Validate ball coordinates
                if 0 <= X <= field_width and 0 <= Y <= field_height:
                    frame_data[frame_num]["ball"] = (X, Y)
                    total_balls_found += 1
                else:
                    print(f"Ball out of field bounds: ({X:.1f}, {Y:.1f}) in frame {frame_num}")
        
        except Exception as e:
            print(f"Error processing result {result_idx}: {e}")
            continue
    
    print(f"Processed data: {len(frame_data)} frames, max frame: {max_frame_num}")
    print(f"Total objects found: {total_players_found} players, {total_balls_found} balls")
    print(f"Team 1 players count: {team1_players_count}")
    print(f"Team 2 players count: {team2_players_count}")
    print(f"Team colors - Team 1: {final_team1_color}, Team 2: {final_team2_color}")
    final_team1_color = (final_team1_color[0], final_team1_color[1], final_team1_color[2])
    final_team2_color = (final_team2_color[0], final_team2_color[1], final_team2_color[2])
    print("INTSSSSSSS")
    print(f"Team colors - Team 1: {final_team1_color}, Team 2: {final_team2_color}")
    
    # Show sample of processed data
    for frame_num in sorted(list(frame_data.keys())[:3]):
        data = frame_data[frame_num]
        print(f"Frame {frame_num}: {len(data['players'])} players, ball: {data['ball'] is not None}")
    
    # Generate frames
    frames_to_process = max(total_frames, max_frame_num + 1) if frame_data else total_frames
    processed_frames = 0
    frames_with_data = 0
    
    for frame_num in range(frames_to_process):
        try:
            # Create minimap with team colors
            minimap = Minimap(
                field_width=field_width,
                field_length=field_height,
                scale=scale,
                ss_factor=ss_factor,
                margin_left=margin_left,
                margin_right=margin_right,
                margin_top=margin_top,
                margin_bottom=margin_bottom,
                team1_color=final_team1_color,  # Pass team colors
                team2_color=final_team2_color
            )
            
            # Draw field lines
            minimap.draw_field_lines(LINES_COORDS)
            
            # Draw objects if frame has data
            if frame_num in frame_data:
                data = frame_data[frame_num]
                if len(data["players"]) > 0 or data["ball"] is not None:
                    frames_with_data += 1
                    
                minimap.draw_objects_with_team_info(data["players"], data["ball"])
            
            # Get final frame
            final_frame = minimap.get_final_minimap()
            
            # Write to video
            out.write(final_frame)
            processed_frames += 1
            
            if processed_frames % 20 == 0:
                print(f"Processed {processed_frames}/{frames_to_process} frames")
                
        except Exception as e:
            print(f"Error creating frame {frame_num}: {e}")
            # Create blank frame
            blank = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            out.write(blank)
            processed_frames += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Visualization complete: {processed_frames} total frames, {frames_with_data} with data")
    print(f"Final team player counts - Team 1: {team1_players_count}, Team 2: {team2_players_count}")
    print(f"Saved to: {field_output_path}")
    return field_output_path


class Minimap:
    """
    Fixed Minimap class with team colors and larger player/ball visualization
    """
    def __init__(self, field_width=105, field_length=68, scale=10, ss_factor=4,
                 margin_left=2, margin_right=2, margin_top=2, margin_bottom=2,
                 team1_color=None, team2_color=None):
        self.field_width = field_width
        self.field_length = field_length
        self.scale = scale
        self.ss_factor = ss_factor
        self.margin_left = margin_left
        self.margin_right = margin_right
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        
        # Store team colors (convert from RGB to BGR for OpenCV)
        if team1_color:
            self.team1_color = (int(team1_color[2]), int(team1_color[1]), int(team1_color[0]))
        else:
            self.team1_color = (0, 0, 255)  # Default red
            
        if team2_color:
            self.team2_color = (int(team2_color[2]), int(team2_color[1]), int(team2_color[0]))
        else:
            self.team2_color = (255, 0, 0)  # Default blue
        
        print(f"Minimap team colors (BGR): Team1={self.team1_color}, Team2={self.team2_color}")
        
        self.minimap_hr, self.m_left, self.m_top = self.create_minimap()

    def create_minimap(self):
        total_width = self.field_width + self.margin_left + self.margin_right
        total_length = self.field_length + self.margin_top + self.margin_bottom
        hr_width = int(total_width * self.scale * self.ss_factor)
        hr_length = int(total_length * self.scale * self.ss_factor)
        
        print(f"Creating minimap: {hr_width} x {hr_length} pixels")
        
        minimap = np.full((hr_length, hr_width, 3), (31, 28, 23), dtype=np.uint8)
        thickness = 2 * self.ss_factor
        cv2.rectangle(minimap, (0, 0), (hr_width - 1, hr_length - 1), (96, 101, 104), thickness)
        return minimap, self.margin_left, self.margin_top

    def draw_field_lines(self, field_lines):
        effective_scale = self.scale * self.ss_factor
        thickness = 2 * self.ss_factor

        for line in field_lines:
            p1, p2 = line
            X1, Y1 = p1[0], p1[1]
            X2, Y2 = p2[0], p2[1]
            pt1 = (int((self.margin_left + X1) * effective_scale),
                   int((self.margin_top + (self.field_length - Y1)) * effective_scale))
            pt2 = (int((self.margin_left + X2) * effective_scale),
                   int((self.margin_top + (self.field_length - Y2)) * effective_scale))
            cv2.line(self.minimap_hr, pt1, pt2, (96, 101, 104), thickness)

        # Draw center circle
        center_field = (self.field_width / 2, self.field_length / 2)
        center_pixel = (int((self.margin_left + center_field[0]) * effective_scale),
                        int((self.margin_top + (self.field_length - center_field[1])) * effective_scale))
        center_circle_radius = 9.15
        radius_pixel = int(center_circle_radius * effective_scale)
        cv2.circle(self.minimap_hr, center_pixel, radius_pixel, (96, 101, 104), thickness, lineType=cv2.LINE_AA)

    def draw_objects_with_team_info(self, player_detections, ball_field_position):
        effective_scale = self.scale * self.ss_factor
        img_height, img_width = self.minimap_hr.shape[:2]
        
        objects_drawn = 0
        
        # Draw players with actual team colors and larger size
        for player in player_detections:
            pos = player["field_position"]
            team = player["team"]
            
            X, Y = pos
            Y_adjusted = Y
            
            # Calculate pixel coordinates
            pt_x = int((self.margin_left + X) * effective_scale)
            pt_y = int((self.margin_top + (self.field_length - Y_adjusted)) * effective_scale)
            
            # Check bounds
            if 0 <= pt_x < img_width and 0 <= pt_y < img_height:
                # Use actual team colors instead of hardcoded ones
                color = self.team1_color if team == 0 else self.team2_color
                print(f"Drawing player at ({X:.1f}, {Y:.1f}) for team {team} with color {color}")
                
                # Make players much larger and more visible
                radius = max(8, int(6 * self.ss_factor))  # Increased from 3 to 8
                
                # Draw player with border for better visibility
                cv2.circle(self.minimap_hr, (pt_x, pt_y), radius, color, -1)  # Filled circle
                cv2.circle(self.minimap_hr, (pt_x, pt_y), radius + 1, (255, 255, 255), 1)  # White border
                
                objects_drawn += 1
            else:
                print(f"Player out of bounds: ({pt_x}, {pt_y}) for team {team}")
        
        # Draw ball with larger size and better visibility
        if ball_field_position:
            X, Y = ball_field_position
            pt_x = int((self.margin_left + X) * effective_scale)
            pt_y = int((self.margin_top + (self.field_length - Y)) * effective_scale)
            
            if 0 <= pt_x < img_width and 0 <= pt_y < img_height:
                # Make ball larger and more visible
                radius = max(6, int(4 * self.ss_factor))  # Increased from 2 to 6
                
                # Draw ball with multiple layers for better visibility
                cv2.circle(self.minimap_hr, (pt_x, pt_y), radius + 2, (0, 0, 0), -1)  # Black background
                cv2.circle(self.minimap_hr, (pt_x, pt_y), radius, (0, 255, 255), -1)  # Yellow ball
                cv2.circle(self.minimap_hr, (pt_x, pt_y), radius, (255, 255, 255), 1)  # White border
                
                objects_drawn += 1
        
        if objects_drawn > 0:
            print(f"Drew {objects_drawn} objects on minimap")
        
        return self.minimap_hr

    def get_final_minimap(self):
        target_width = int((self.field_width + self.margin_left + self.margin_right) * self.scale)
        target_height = int((self.field_length + self.margin_top + self.margin_bottom) * self.scale)
        minimap_final = cv2.resize(self.minimap_hr, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return minimap_final
