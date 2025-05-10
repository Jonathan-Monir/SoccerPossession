def detections_to_dicts(ball_dets, player_dets):
    """
    Convert Norfair Detection objects into plain dicts
    with field_position (and class_id for players).
    """
    ball_list = []
    for b in ball_dets or []:
        # centroid of the points
        pixel = b.points.mean(axis=0).tolist()
        field = coords_to_dict(pixel)["field_position"]
        ball_list.append({"field_position": field})
    player_list = []
    for p in player_dets or []:
        pixel = p.points.mean(axis=0).tolist()
        field = coords_to_dict(pixel)["field_position"]
        # assume your Detection has .class_id
        player_list.append({"field_position": field, "class_id": p.class_id})
    return ball_list, player_list

# -----------------------------------------------------------------------------
# Possession calculation (updated)
# -----------------------------------------------------------------------------

def CalculatePossession(
    data,
    yardTL=[0.0, 68.0],
    yardTR=[105.0, 68.0],
    yardBL=[0.0, 0.0],
    yardBR=[105.0, 0.0],
):
    frames = 0
    framesT1 = framesT2 = 0
    cumulative_possessions = []
    team_possession_list = []
    prevBall = None

    for i, entry in enumerate(data):
        frame, raw_ball, raw_players = entry

        # 1) convert Norfair detections -> dicts
        ball_list, players_list = detections_to_dicts(raw_ball, raw_players)

        # 2) validate & handle continuity
        ball = ValidateBall(ball_list)
        players = ValidatePlayers(players_list)
        ball = HandleBallWithValidation(ball, prevBall, yardTL, yardTR, yardBL, yardTL[1])

        # 3) possession logic
        if ball and isinstance(ball, dict):
            owner = GetClosestTeam(ball["field_position"], players or [])
            if owner == 1:
                framesT1 += 1; frames += 1
            elif owner == 2:
                framesT2 += 1; frames += 1
            team_possession_list.append(owner)
        else:
            team_possession_list.append(None)

        cumulative_possessions.append(
            GetPossessionPercentage(frames, framesT1, framesT2)
        )
        prevBall = ball if isinstance(ball, dict) else None

    return cumulative_possessions, team_possession_list

def ValidateBall(ball):
    if isinstance(ball, list):
        for det in ball:
            if isinstance(det, dict) and "field_position" in det:
                return det
        return None
    if isinstance(ball, dict) and "field_position" in ball:
        return ball
    return None

def ValidatePlayers(players):
    if isinstance(players, list):
        return [
            p for p in players
            if isinstance(p, dict)
            and "field_position" in p
            and "class_id" in p
            and p["class_id"] in {1,2}
        ]
    return []

def HandleBallWithValidation(current_ball, prev_ball, tl, tr, bl, max_y):
    valid_current = isinstance(current_ball, dict) and "field_position" in current_ball
    valid_prev    = isinstance(prev_ball, dict) and "field_position" in prev_ball

    if not valid_current:
        if valid_prev and IsInBounds(prev_ball["field_position"], tl, tr, bl, max_y):
            return prev_ball
        return None

    pos = current_ball["field_position"]
    if IsInBounds(pos, tl, tr, bl, max_y):
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
    min_dist = float("inf")
    team = None
    for p in players:
        try:
            p_pos = p["field_position"]
            d = ((p_pos[0]-ball_pos[0])**2 + (p_pos[1]-ball_pos[1])**2)**0.5
            if d < min_dist:
                min_dist = d
                team = p["class_id"]
        except:
            continue
    return team if team in {1,2} else None

def GetPossessionPercentage(frames, f1, f2):
    if frames == 0:
        return {"possT1": 0.0, "possT2": 0.0}
    return {"possT1": round(f1/frames*100,1), "possT2": round(f2/frames*100,1)}
