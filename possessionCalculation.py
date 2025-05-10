import time
import warnings

from yolo_norfair import yolo_to_norfair_detections
from Transformation.utils.utils_heatmap import coords_to_dict
from tracker import process_video
from cluster_time_improve import main_multi_frame
from transformation import process_field_transformation
from videoDrawImprove import draw_bounding_boxes_on_frames
from make_vid import vid

warnings.filterwarnings("ignore")

def measure_time(func, *args, process_name="Process"):
    start = time.time()
    out = func(*args)
    print(f"{process_name} took {time.time()-start:.3f}s")
    return out

def detections_to_dicts(ball_dets, player_dets):
    balls, players = [], []
    for b in ball_dets or []:
        pixel = b.points.mean(axis=0).tolist()
        fld = coords_to_dict(pixel)["field_position"]
        balls.append({"field_position": fld})
    for p in player_dets or []:
        pixel = p.points.mean(axis=0).tolist()
        fld = coords_to_dict(pixel)["field_position"]
        # ensure your Detection has .class_id
        players.append({"field_position": fld, "class_id": p.class_id})
    return balls, players

def CalculatePossession(data, yardTL, yardTR, yardBL, yardBR):
    total = 0
    valid_ball = 0
    owner1 = 0
    owner2 = 0
    skipped = 0

    frames = f1 = f2 = 0
    poss_list = []
    team_seq = []
    prev = None

    for i, (frame, raw_ball, raw_players) in enumerate(data):
        total += 1
        balls, players = detections_to_dicts(raw_ball, raw_players)
        ball = ValidateBall(balls)
        players = ValidatePlayers(players)
        # use bottom-right Y for max_y
        ball = HandleBallWithValidation(ball, prev, yardTL, yardTR, yardBL, yardBR[1])

        if isinstance(ball, dict):
            valid_ball += 1
            owner = GetClosestTeam(ball["field_position"], players)
            if owner == 1:
                f1 += 1; frames += 1; owner1 += 1
            elif owner == 2:
                f2 += 1; frames += 1; owner2 += 1
            else:
                skipped += 1
            team_seq.append(owner)
        else:
            skipped += 1
            team_seq.append(None)

        poss_list.append(GetPossessionPercentage(frames, f1, f2))
        prev = ball if isinstance(ball, dict) else None

    # summary
    print("\n=== POSSESSION DEBUG SUMMARY ===")
    print(f"Total frames:        {total}")
    print(f"Frames with ball:    {valid_ball}")
    print(f" Team 1 touches:     {owner1}")
    print(f" Team 2 touches:     {owner2}")
    print(f" Frames counted:     {frames}")
    print(f" Frames skipped:     {skipped}")
    print("================================\n")

    return poss_list, team_seq

def ValidateBall(ball):
    if isinstance(ball, list):
        for b in ball:
            if isinstance(b, dict):
                return b
        return None
    return ball if isinstance(ball, dict) else None

def ValidatePlayers(players):
    return [p for p in (players or []) if p.get("class_id") in (1,2)]

def HandleBallWithValidation(cur, prev, tl, tr, bl, max_y):
    valid_cur = isinstance(cur, dict)
    valid_prev = isinstance(prev, dict)
    if not valid_cur:
        return prev if valid_prev and IsInBounds(prev["field_position"], tl, tr, bl, max_y) else None
    pos = cur["field_position"]
    if IsInBounds(pos, tl, tr, bl, max_y):
        return cur
    return prev if valid_prev and IsInBounds(prev["field_position"], tl, tr, bl, max_y) else None

def IsInBounds(pos, tl, tr, bl, max_y):
    try:
        return tl[0] <= pos[0] <= tr[0] and bl[1] <= pos[1] <= max_y
    except:
        return False

def GetClosestTeam(ball_pos, players):
    best, team = float("inf"), None
    for p in players:
        d = ((p["field_position"][0]-ball_pos[0])**2 + (p["field_position"][1]-ball_pos[1])**2)**0.5
        if d < best:
            best, team = d, p["class_id"]
    return team

def GetPossessionPercentage(frames, t1, t2):
    if frames == 0: return {"possT1": 0.0, "possT2": 0.0}
    return {"possT1": round(t1/frames*100,1), "possT2": round(t2/frames*100,1)}
