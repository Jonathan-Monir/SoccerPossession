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
        frame, ball, players = entry  # Keep original unpacking order
        ball = entry['ball']
        players = entry['players']

        # Validate ball detection
        ball = ValidateBall(ball)
        players = ValidatePlayers(players)

        print(f"ball::: {ball}")
        print(f"players::: {players}")
        # Handle ball continuity with type checks
        ball = HandleBallWithValidation(ball, prevBall, yardTL, yardTR, yardBL, yardTL[1])


        print(f"ball with valid::: {ball}")
        print(f"players with valid::: {players}")

        # Process possession only with valid data
        if ball and isinstance(ball, dict) and "field_position" in ball:
            owner = GetClosestTeam(ball["field_position"], players)
            if owner == 1:
                framesT1 += 1
                frames += 1
            elif owner == 2:
                framesT2 += 1
                frames += 1
            team_possession_list.append(owner)
        else:
            team_possession_list.append(None)
	
	print("frames" + frames)
	print("t1" + framesT1)
	print("t2" + framesT2)

        cumulative_possessions.append(
            GetPossessionPercentage(frames, framesT1, framesT2)
        )
        prevBall = ball if isinstance(ball, dict) else None

    return cumulative_possessions, team_possession_list

def ValidateBall(ball):
    """Ensure ball is a valid detection dict"""
    if isinstance(ball, list):
        # Return first valid dict detection
        for det in ball:
            if isinstance(det, dict) and "field_position" in det:
                return det
        return None
    if isinstance(ball, dict) and "field_position" in ball:
        return ball
    return None

def ValidatePlayers(players):
    """Filter valid player dicts"""
    if isinstance(players, list):
        return [
            p for p in players
            if isinstance(p, dict) and 
            "field_position" in p and 
            "class_id" in p and 
            p["class_id"] in {1, 2}
        ]
    return None

def HandleBallWithValidation(current_ball, prev_ball, tl, tr, bl, max_y):
    """Safe ball continuity handling"""
    # Validate current ball
    valid_current = (
        isinstance(current_ball, dict) and 
        "field_position" in current_ball
    )
    
    # Validate previous ball
    valid_prev = (
        isinstance(prev_ball, dict) and 
        "field_position" in prev_ball
    )

    # Use previous ball if current is invalid
    if not valid_current:
        if valid_prev and IsInBounds(prev_ball["field_position"], tl, tr, bl, max_y):
            return prev_ball
        return None

    # Check current ball position
    current_pos = current_ball["field_position"]
#     print(f"is valid: {valid_current}, infield: ball{IsInBounds(current_pos,tl,tr,bl,max_y)}")
    if IsInBounds(current_pos, tl, tr, bl, max_y):
        return current_ball

    # Fallback to valid previous ball
    if valid_prev and IsInBounds(prev_ball["field_position"], tl, tr, bl, max_y):
        return prev_ball
    
    return None

def IsInBounds(pos, tl, tr, bl, max_y):
    print(f"pos: {pos}")
    print(f"tl: {tl}")
    print(f"tr: {tr}")
    print(f"bl: {bl}")
    print(f"my: {max_y}")
    """Safe coordinate validation"""
    try:
        return (
            tl[0] <= pos[0] <= tr[0] and 
            bl[1] <= pos[1] <= max_y
        )
    except (TypeError, IndexError):
        return False

def GetClosestTeam(ball_pos, players):
    """Find nearest player to ball"""
    min_dist = float("inf")
    closest_team = None
    
    for p in players:
        if not isinstance(p, dict):
            continue
        try:
            p_pos = p["field_position"]
            dist = ((p_pos[0]-ball_pos[0])**2 + (p_pos[1]-ball_pos[1])**2)**0.5
            if dist < min_dist:
                min_dist = dist
                closest_team = p.get("class_id")
        except (KeyError, TypeError):
            continue
    
    return closest_team if closest_team in {1, 2} else None

def GetPossessionPercentage(frames, f1, f2):
    return {
        "possT1": round(f1/(f1+f2)*100, 1) if frames else 0.0,
        "possT2": round(f2/(f1+f2)*100, 1) if frames else 0.0
    }
