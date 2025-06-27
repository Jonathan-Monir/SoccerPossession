import io
import subprocess
import boto3
from botocore.exceptions import NoCredentialsError
import json
import logging
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4
import uuid
import base64
import cv2
import boto3
from botocore.exceptions import NoCredentialsError
import sys
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from botocore.exceptions import ClientError
import tempfile
from tracker import process_video
from cluster import main_multi_frame
from transformation import process_field_transformation
from possessionCalculation import CaculatePossession
import warnings
from videoDraw import draw_bounding_boxes_on_frames, save_video_from_frames
from urllib.parse import quote

import math
teams_jersey_colors = {
  "FC Barcelona": [[0, 0, 139], [178, 34, 34], [255, 215, 0]],
  "Real Madrid": [[255, 255, 255], [0, 0, 139], [128, 128, 128]],
  "Manchester United": [[220, 20, 60], [255, 255, 255], [0, 0, 0]],
  "Manchester City": [[135, 206, 235], [0, 0, 128], [0, 0, 0]],
  "Liverpool": [[200, 16, 46], [255, 255, 255], [0, 100, 0]],
  "Chelsea": [[0, 0, 205], [255, 255, 0], [255, 255, 255]],
  "Arsenal": [[255, 0, 0], [255, 255, 255], [255, 215, 0], [0, 0, 0]],
  "Tottenham Hotspur": [[255, 255, 255], [0, 0, 128], [135, 206, 235]],
  "Bayern Munich": [[220, 20, 60], [255, 255, 255], [0, 0, 0]],
  "Borussia Dortmund": [[255, 215, 0], [0, 0, 0], [255, 69, 0]],
  "PSG": [[0, 0, 139], [220, 20, 60], [255, 192, 203]],
  "Juventus": [[255, 255, 255], [0, 0, 0], [0, 0, 255]],
  "AC Milan": [[139, 0, 0], [0, 0, 0], [255, 255, 255]],
  "Inter Milan": [[0, 0, 205], [0, 0, 0], [255, 255, 255]],
  "AS Roma": [[128, 0, 0], [255, 255, 0], [255, 255, 255]],
  "Napoli": [[135, 206, 235], [0, 0, 139], [0, 0, 0]],
  "Atletico Madrid": [[255, 0, 0], [255, 255, 255], [0, 0, 139]],
  "Sevilla": [[255, 255, 255], [220, 20, 60], [0, 0, 0]],
  "Ajax": [[255, 255, 255], [220, 20, 60], [0, 0, 0]],
  "FC Porto": [[0, 0, 205], [255, 255, 255], [255, 215, 0]],
  "Benfica": [[255, 0, 0], [255, 255, 255], [0, 0, 0]],
  "Sporting CP": [[0, 128, 0], [255, 255, 255], [0, 0, 0]],
  "Galatasaray": [[220, 20, 60], [255, 215, 0], [0, 0, 0]],
  "Fenerbahçe": [[255, 255, 0], [0, 0, 139], [255, 255, 255]],
  "Celtic": [[0, 128, 0], [255, 255, 255], [255, 255, 0]],
  "Rangers": [[0, 0, 205], [255, 255, 255], [255, 0, 0]],
  "Lyon": [[255, 255, 255], [0, 0, 139], [255, 0, 0]],
  "Marseille": [[255, 255, 255], [135, 206, 250], [0, 0, 128]],
  "Bayer Leverkusen": [[220, 20, 60], [0, 0, 0], [192, 192, 192]],
  "RB Leipzig": [[255, 255, 255], [255, 0, 0], [0, 0, 0]],
  "Real Betis": [[0, 128, 0], [255, 255, 255], [0, 100, 0]],
  "Elche CF": [[255, 255, 255], [0, 128, 0], [0, 0, 0]]
}

import math

def color_distance(c1, c2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

def assign_best_unique_colors(color1, color2, team1, team2, teams_colors):
    if team1 not in teams_colors or team2 not in teams_colors:
        raise ValueError("One or both teams not found in jersey color dictionary.")

    combinations = []
    for color, label in zip([color1, color2], ['color1', 'color2']):
        for team in [team1, team2]:
            for jersey_color in teams_colors[team]:
                combinations.append({
                    'color_label': label,
                    'team': team,
                    'jersey_color': jersey_color,
                    'input_color': color,
                    'distance': color_distance(color, jersey_color)
                })

    best_pair = None
    best_total_distance = float('inf')

    for c1 in combinations:
        for c2 in combinations:
            if c1['color_label'] == c2['color_label']:
                continue
            if c1['team'] == c2['team']:
                continue

            total_dist = c1['distance'] + c2['distance']
            if total_dist < best_total_distance:
                print("**********\nCOLOR 1\n*********")
                print(f"color 1 = {c1['input_color']} in distance = {c1['distance']} from actual = {c1['jersey_color']}")
                print("**********\nCOLOR 2\n*********")
                print(f"and color 2 = {c2['input_color']} in distance = {c2['distance']} from actual = {c2['jersey_color']}")
                best_total_distance = total_dist
                best_pair = {
                    c1['team']: {'input_color': c1['input_color'],
                                 'jersey_color': c1['jersey_color']},
                    c2['team']: {'input_color': c2['input_color'],
                                 'jersey_color': c2['jersey_color']}
                }
            
    print("**********\nFinally\n*********")
    print(f"best_pair = {best_pair}")

    return best_pair


warnings.filterwarnings("ignore")

def measure_time(func, *args, process_name="Process"):
    """Helper function to measure execution time."""
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{process_name} completed in {elapsed_time:.4f} seconds")
    return result

logging.basicConfig(level=logging.INFO)

# Kaggle-specific paths
WORKING_DIR = r"\output"
PROCESSED_VIDEOS_DIR = os.path.join(WORKING_DIR, "processed_videos")
os.makedirs(PROCESSED_VIDEOS_DIR, exist_ok=True)

print(f"Working directory: {WORKING_DIR}")
print(f"Processed videos directory: {PROCESSED_VIDEOS_DIR}")

app = FastAPI()

# Mount static files with the correct Kaggle path
app.mount("/processed_videos", StaticFiles(directory=PROCESSED_VIDEOS_DIR), name="processed_videos")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# إعداد متغيرات S3 (يجب وضعها في environment variables)
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, AWS_BUCKET_NAME]):
    raise Exception("يجب تعيين متغيرات البيئة الخاصة بـ AWS: AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, AWS_BUCKET_NAME")

s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )


def convert_with_ffmpeg(input_path: str) -> str:
    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    print(f"temp_output = {temp_output}")
    output_path = temp_output.name
    temp_output.close()
    # print(f"output_path = {output_path}")
    command = [
        'ffmpeg',
        '-y',  # <--- allow overwrite
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-movflags', '+faststart',
        output_path
    ]
    print(f"command = {command}")
    try:
        subprocess.run(command, check=True)
        print(f"output_path = {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error: {e}")



# دالة رفع الفيديو إلى S3
def upload_to_s3(file_path, s3_key):
    if not os.path.exists(file_path):
        raise Exception(f"File does not exist: {file_path}")
    try:
        print(f"Uploading to S3 bucket: {AWS_BUCKET_NAME}, key: {s3_key}")
        s3.upload_file(file_path, AWS_BUCKET_NAME, s3_key, ExtraArgs={'ContentType': 'video/mp4', 'ContentDisposition': 'attachment'})
        # إنشاء presigned URL صالح لمدة 24 ساعة
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': AWS_BUCKET_NAME, 'Key': s3_key},
            ExpiresIn=86400
        )
        print(f"Generated presigned URL: {url}")
        return url
        # file_obj.seek(0)
        # s3.upload_fileobj(file_obj, bucket, object_name)
        # print(f"Uploaded {object_name} to S3.")
        # return f"https://{bucket}.s3.amazonaws.com/{object_name}"
    except NoCredentialsError:
        raise Exception("AWS credentials not found")
    except Exception as e:
        raise Exception(f"S3 upload error: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "FastAPI server running on Kaggle", "working_dir": WORKING_DIR}

@app.get("/debug/files")
async def list_files():
    """Debug endpoint to list files in processed_videos directory"""
    try:
        files = os.listdir(PROCESSED_VIDEOS_DIR)
        return {
            "processed_videos_dir": PROCESSED_VIDEOS_DIR,
            "files": files,
            "file_count": len(files)
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/process-video/")
async def processVideo(
    video: UploadFile = File(...),
    fps: int = Form(30),
    yolo_model_path: str = r"resources/yolo8.pt",
    team1: str = Form("Team A"),
    team2: str = Form("Team B")):

    # Generate a unique filename for the processed video
    unique_id = uuid.uuid4().hex
    output_filename = f"output_{unique_id}.mp4"
    output_video_path = os.path.join(PROCESSED_VIDEOS_DIR, output_filename)

    # Save uploaded video to a temporary file in working directory
    temp_video_path = os.path.join(WORKING_DIR, f"temp_{uuid4().hex}.mp4")
    
    print(f"Saving temp video to: {temp_video_path}")
    print(f"Output video will be saved to: {output_video_path}")
    
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    try:
        # TRACKING
        results = measure_time(process_video, yolo_model_path, temp_video_path, fps, process_name="Tracking")
    
        print("FINISHING TRACKING")
        # CLUSTERING
        results_with_class_ids, team1_color, team2_color = measure_time(main_multi_frame, results, process_name="Clustering")
        # print(f"FINISHING CLUSTERING = {results_with_class_ids}")
        print(f"team1_color = {team1_color}")
        print(f"team2_color = {team2_color}")

        final_color_teams = assign_best_unique_colors(team1_color, team2_color, team1, team2,teams_jersey_colors)
        
        print(f"FINISHING ASSIGNING COLORS = {final_color_teams}")
        if final_color_teams[team1]['input_color'] == team1_color:
            final_team1_color = final_color_teams[team1]['jersey_color']
            final_team2_color = final_color_teams[team2]['jersey_color']
        else:
            final_team2_color = final_color_teams[team1]['jersey_color']
            final_team1_color = final_color_teams[team2]['jersey_color']
            x = team1
            team1 = team2
            team2 = x
            
        print("***********************************************************************")
        print(f"final_color_teams[{team1}] = {final_team1_color}")
        print(f"final_color_teams[{team2}] = {final_team2_color}")
        print("***********************************************************************")
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
        print("FINISHING FIELD TRANSFORMATION")
        # POSSESSION CALCULATION
        yardTL, yardTR, yardBL, yardBR = [29.0, 17.0], [45.5, 17.0], [29.0, 26.0], [45.5, 26.0]
        poss, team_poss_list = measure_time(CaculatePossession, results, yardTL, yardTR, yardBL, yardBR, process_name="Possession Calculation")
        print("FINISHING CALCULATING POSSESSION")
        # GENERATE VISUALIZATION
        visualize = draw_bounding_boxes_on_frames(results_with_class_ids, final_team1_color, final_team2_color, team_poss_list)

        # SAVE PROCESSED VIDEO
        print(f"Saving processed video to: {output_video_path}")
        save_video_from_frames(visualize, output_path=output_video_path)
        
        # Verify file was created
        if not os.path.exists(output_video_path):
            raise Exception(f"Failed to create output video at {output_video_path}")

        print(f"BEFORE CONVERTING Output video path = {output_video_path}")
        output_video_path = convert_with_ffmpeg(output_video_path)
        print(f"AFTER CONVERTING Output video path = {output_video_path}")
        file_size = os.path.getsize(output_video_path)
        print(f"Output video created successfully. Size: {file_size} bytes")
        
        # رفع الفيديو إلى S3
        s3_key = f"processed_videos/{output_filename}"
        s3_url = upload_to_s3(output_video_path, s3_key)
        
        # حذف الملف المحلي بعد الرفع
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
            
        # video_url = f"https://labrador-fresh-eminently.ngrok-free.app/processed_videos/{output_filename}"
        video_url = s3_url
        print(f"S3 Video URL: {video_url}")
        print(f"team1_color = {team1_color} and team2_color = {team2_color}")
        print(f"final_team1_color = {final_team1_color} and final_team2_color = {final_team2_color}")
        return JSONResponse(
            status_code=200,
            content={
                "message": "Video processed successfully",
                "possession": poss,
                "videoUrl": video_url,
                "videoId": unique_id,
                "team1": team1,
                "team2": team2,
                "team1_colorR": int(final_team1_color[0]),
                "team1_colorG": int(final_team1_color[1]),
                "team1_colorB": int(final_team1_color[2]),
                "team2_colorR": int(final_team2_color[0]),
                "team2_colorG": int(final_team2_color[1]),
                "team2_colorB": int(final_team2_color[2])
            }
        )
    except Exception as e:
        # Clean up files if error occurs
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        print(f"Error during processing: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    finally:
        # Clean up temp file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            print(f"Cleaned up temp file: {temp_video_path}")

@app.get("/download-video/{video_id}")
async def download_video(video_id: str):
    """Download endpoint for processed videos from S3"""
    video_filename = f"output_{video_id}.mp4"
    s3_key = f"processed_videos/{video_filename}"
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        # تحقق من وجود الملف في S3
        s3.head_object(Bucket=AWS_BUCKET_NAME, Key=s3_key)
        # أنشئ presigned url
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': AWS_BUCKET_NAME, 'Key': s3_key},
            ExpiresIn=86400
        )
        return {"download_url": url}
    except s3.exceptions.NoSuchKey:
        return JSONResponse(
            status_code=404,
            content={"error": "Video not found in S3", "key": s3_key}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/direct-stream/{video_id}")
def get_direct_stream(video_id: str):
    try:
        output_filename = f"output_{video_id}.mp4"
        video_key = f"processed_videos/{output_filename}"
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': AWS_BUCKET_NAME, 'Key': video_key},
            ExpiresIn=3600  # 1 hour expiration
        )
        return RedirectResponse(url)
    except ClientError as e:
        raise HTTPException(status_code=404, detail=f"video_key = {video_key} and AWS_BUCKET_NAME = {AWS_BUCKET_NAME}")


# # Launch FastAPI in a thread
# def run():
#     uvicorn.run(app, host="127.0.0.1", port=7468)