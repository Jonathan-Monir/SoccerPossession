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
        frame, ball, players = entry

        # Normalize empty lists to None and handle invalid types
        if isinstance(players, list) and len(players) == 0:
            players = None
        if isinstance(ball, list) and len(ball) == 0:
            ball = None
        # Ensure ball is a dictionary if not None
        if ball is not None:
            if isinstance(ball, list):
                if len(ball) == 1:
                    ball = ball[0]  # Extract single detection from list
                else:
                    ball = None  # Invalid list length after processing
            if not isinstance(ball, dict):
                ball = None  # Ensure ball is a dictionary

        # Handle cases with no players or initial frame with no ball
        if players is None or (i == 0 and ball is None):
            cumulative_possessions.append(
                GetPossessionPercentage(frames, framesT1, framesT2)
            )
            team_possession_list.append(None)
            continue

        # Track previous ball position
        prev_pos = prevBall["field_position"] if prevBall else None

        # Select best ball from multiple detections if possible
        if isinstance(ball, list) and len(ball) > 1:
            if (
                prevBall
                and yardTL[0] <= prev_pos[0] <= yardTR[0]
                and yardBL[1] <= prev_pos[1] <= yardTL[1]
            ):
                best = ball[0]
                best_d = DistanceBetweenObjects(best["field_position"], prev_pos)
                for det in ball[1:]:
                    pos = det["field_position"]
                    if (
                        yardTL[0] <= pos[0] <= yardTR[0]
                        and yardBL[1] <= pos[1] <= yardTL[1]
                    ):
                        d = DistanceBetweenObjects(pos, prev_pos)
                        if d < best_d:
                            best_d, best = d, det
                ball = best
            else:
                ball = ball[0]

        # Fallback to previous ball if current is None and within bounds
        if ball is None and i > 0:
            if prevBall and prevBall.get("field_position"):
                prev_pos = prevBall["field_position"]
                if (
                    yardTL[0] <= prev_pos[0] <= yardTR[0]
                    and yardBL[1] <= prev_pos[1] <= yardTL[1]
                ):
                    ball = prevBall

        # Validate ball is a dictionary before proceeding
        if ball is not None and not isinstance(ball, dict):
            ball = None

        if ball is None:
            cumulative_possessions.append(
                GetPossessionPercentage(frames, framesT1, framesT2)
            )
            team_possession_list.append(None)
            prevBall = ball
            continue

        # Calculate distances from players to the ball
        distances = {"T1": [], "T2": []}
        ball_pos = ball.get("field_position")
        if not ball_pos:
            cumulative_possessions.append(
                GetPossessionPercentage(frames, framesT1, framesT2)
            )
            team_possession_list.append(None)
            prevBall = ball
            continue

        for p in players:
            if not isinstance(p, dict):
                continue  # Skip invalid player entries
            p_pos = p.get("field_position")
            p_team = p.get("class_id")
            if not p_pos or p_team not in [1, 2]:
                continue
            d = DistanceBetweenObjects(p_pos, ball_pos)
            if p_team == 1:
                distances["T1"].append(d)
            else:
                distances["T2"].append(d)

        prevBall = ball

        # Determine possession based on closest player
        if not distances["T1"] and not distances["T2"]:
            cumulative_possessions.append(
                GetPossessionPercentage(frames, framesT1, framesT2)
            )
            team_possession_list.append(None)
            continue

        owner = None
        if not distances["T1"]:
            framesT2 += 1
            owner = 2
        elif not distances["T2"]:
            framesT1 += 1
            owner = 1
        else:
            min_t1 = min(distances["T1"])
            min_t2 = min(distances["T2"])
            if min_t1 < min_t2:
                framesT1 += 1
                owner = 1
            else:
                framesT2 += 1
                owner = 2

        frames += 1
        team_possession_list.append(owner)
        cumulative_possessions.append(
            GetPossessionPercentage(frames, framesT1, framesT2)
        )

    return cumulative_possessions, team_possession_list

def DistanceBetweenObjects(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def GetPossessionPercentage(frames, f1, f2):
    if frames == 0:
        return {"possT1": 0, "possT2": 0}
    return {"possT1": (f1 / frames) * 100, "possT2": (f2 / frames) * 100}