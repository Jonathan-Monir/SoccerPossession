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
        # Properly unpack the entry with correct order
        frame, players, ball = entry  # Fixed order: (frame, players, ball)

        # Normalize and validate data types
        ball = ValidateBallDetection(ball)
        players = ValidatePlayers(players)

        # Handle initial frame with no valid ball
        if i == 0 and ball is None:
            cumulative_possessions.append(GetPossessionPercentage(frames, framesT1, framesT2))
            team_possession_list.append(None)
            continue

        # Track ball position with continuity checks
        ball = HandleBallContinuity(ball, prevBall, yardTL, yardTR, yardBL, yardTL[1])
        
        # Calculate possession only if we have valid data
        if ball and players:
            owner = DeterminePossession(ball, players)
            UpdateCounters(owner)
            team_possession_list.append(owner)
        else:
            team_possession_list.append(None)

        # Update cumulative stats
        cumulative_possessions.append(GetPossessionPercentage(frames, framesT1, framesT2))
        prevBall = ball

    return cumulative_possessions, team_possession_list

# Helper functions
def ValidateBallDetection(ball):
    if isinstance(ball, list):
        if len(ball) == 0:
            return None
        # Take first detection that's a dictionary with coordinates
        for detection in ball:
            if isinstance(detection, dict) and 'field_position' in detection:
                return detection
        return None
    if isinstance(ball, dict) and 'field_position' in ball:
        return ball
    return None

def ValidatePlayers(players):
    if isinstance(players, list):
        return [p for p in players if isinstance(p, dict) and 
                'field_position' in p and 
                'class_id' in p and 
                p['class_id'] in {1, 2}]
    return None

def HandleBallContinuity(current_ball, prevBall, yardTL, yardTR, yardBL, maxY):
    if not current_ball:
        # Use previous ball if it's within valid field coordinates
        if prevBall and IsInField(prevBall['field_position'], yardTL, yardTR, yardBL, maxY):
            return prevBall
        return None
    
    pos = current_ball['field_position']
    if IsInField(pos, yardTL, yardTR, yardBL, maxY):
        return current_ball
    return prevBall if prevBall and IsInField(prevBall['field_position'], yardTL, yardTR, yardBL, maxY) else None

def IsInField(pos, tl, tr, bl, maxY):
    return tl[0] <= pos[0] <= tr[0] and bl[1] <= pos[1] <= maxY

def DeterminePossession(ball, players):
    ball_pos = ball['field_position']
    min_dist = float('inf')
    closest_team = None
    
    for player in players:
        if 'field_position' not in player or 'class_id' not in player:
            continue
            
        dist = DistanceBetweenObjects(player['field_position'], ball_pos)
        if dist < min_dist:
            min_dist = dist
            closest_team = player['class_id']
    
    return closest_team if closest_team in {1, 2} else None

def UpdateCounters(owner, frames, framesT1, framesT2):
    if owner == 1:
        framesT1 += 1
        frames += 1
    elif owner == 2:
        framesT2 += 1
        frames += 1
    return frames, framesT1, framesT2

def DistanceBetweenObjects(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def GetPossessionPercentage(frames, f1, f2):
    if frames == 0:
        return {"possT1": 0.0, "possT2": 0.0}
    return {
        "possT1": round((f1 / frames) * 100, 1),
        "possT2": round((f2 / frames) * 100, 1)
    }