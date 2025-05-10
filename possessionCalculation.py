import time
import warnings

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
        players.append({"field_position": fld, "class_id": p.class_id})
    return balls, players

def CalculatePossession(data):
    yardTL, yardTR, yardBL, yardBR = [29.0, 17.0], [45.5, 17.0], [29.0, 26.0], [45.5, 26.0]
    total = valid_ball = owner1 = owner2 = skipped = 0
    frames = f1 = f2 = 0
    poss_list, team_seq = [], []
    prev = None

    for i, entry in enumerate(data):
        # your original unpacking
        frame, ball, players = entry  
        ball = entry['ball']
        players = entry['players']

        total += 1
        balls, players = detections_to_dicts(ball, players)
        ball = ValidateBall(balls)
        players = ValidatePlayers(players)
        # use bottom-right Y for max_y
        ball = HandleBallWithValidation(ball, prev, yardTL, yardTR, yardBL, yardBR[1])

        print(f"ball::: {ball}")
        print(f"players::: {players}")
        # Handle ball continuity with type checks
        ball = HandleBallWithValidation(ball, prev, yardTL, yardTR, yardBL, yardTL[1])
        print(f"ball with valid::: {ball}")
        print(f"players with valid::: {players}")

        # Process possession only with valid data
        if ball and isinstance(ball, dict) and "field_position" in ball:
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

def HandleBallWithValidation(current_ball, prev_ball, tl, tr, bl, max_y):
    """Safe ball continuity handling"""
    valid_current = isinstance(current_ball, dict) and "field_position" in current_ball
    valid_prev    = isinstance(prev_ball, dict)   and "field_position" in prev_ball

    if not valid_current:
        if valid_prev and IsInBounds(prev_ball["field_position"], tl, tr, bl, max_y):
            return prev_ball
        return None

    current_pos = current_ball["field_position"]
    if IsInBounds(current_pos, tl, tr, bl, max_y):
        return current_ball

    if valid_prev and IsInBounds(prev_ball["field_position"], tl, tr, bl, max_y):
        return prev_ball
    
    return None

def IsInBounds(pos, tl, tr, bl, max_y):
    try:
        return tl[0] <= pos[0] <= tr[0] and bl[1] <= pos[1] <= max_y
    except:
        return False

def GetClosestTeam(ball_pos, players):
    best, team = float("inf"), None
    for p in players:
        d = ((p["field_position"][0]-ball_pos[0])**2 + 
             (p["field_position"][1]-ball_pos[1])**2)**0.5
        if d < best:
            best, team = d, p["class_id"]
    return team

def GetPossessionPercentage(frames, f1, f2):
    if frames == 0:
        return {"possT1": 0.0, "possT2": 0.0}
    return {
        "possT1": round(f1/frames*100, 1),
        "possT2": round(f2/frames*100, 1)
    }
