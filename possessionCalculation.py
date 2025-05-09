def CalculatePossession(
    data,
    yardTL=[0.0, 68.0],
    yardTR=[105.0, 68.0],
    yardBL=[0.0, 0.0],
    yardBR=[105.0, 0.0],
):
    # total frames counted, plus per-team counters
    frames = 0
    framesT1 = framesT2 = 0

    cumulative_possessions = []
    team_possession_list = []
    prevBall = None

    for i, entry in enumerate(data):
        # Unpack exactly (frame, ball, players)
        # — drop the frame index or image if you don't need it
        frame, ball, players = entry

        # Normalize empty‐list cases → treat as “no detection”
        if isinstance(players, list) and len(players) == 0:
            players = None
        if isinstance(ball, list) and len(ball) == 0:
            ball = None

        # If there are no players, or (first frame & no ball), we can only append the last %
        if players is None or (i == 0 and ball is None):
            cumulative_possessions.append(
                GetPossessionPercentage(frames, framesT1, framesT2)
            )
            team_possession_list.append(None)
            continue

        # Previous ball position for distance‐based filtering
        prev_pos = prevBall["field_position"] if prevBall else None

        # If multiple ball detections, pick the one closest to prevPos (if in bounds)
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

        # If we still have no ball (and it’s not frame 0), maybe reuse prevBall
        if ball is None and i > 0:
            if (
                prevBall
                and yardTL[0] <= prev_pos[0] <= yardTR[0]
                and yardBL[1] <= prev_pos[1] <= yardTL[1]
            ):
                ball = prevBall
            else:
                cumulative_possessions.append(
                    GetPossessionPercentage(frames, framesT1, framesT2)
                )
                team_possession_list.append(None)
                continue

        # Compute distances from each player to the ball
        distances = {"T1": [], "T2": []}
        for p in players:
            d = DistanceBetweenObjects(
                p["field_position"], ball["field_position"]
            )
            if p["class_id"] == 1:
                distances["T1"].append(d)
            elif p["class_id"] == 2:
                distances["T2"].append(d)

        prevBall = ball  # for the next frame’s filtering

        # If neither team has any players near the ball, skip
        if not distances["T1"] and not distances["T2"]:
            cumulative_possessions.append(
                GetPossessionPercentage(frames, framesT1, framesT2)
            )
            team_possession_list.append(None)
            continue

        # Decide who has possession this frame
        if not distances["T1"]:
            framesT2 += 1
            owner = 2
        elif not distances["T2"]:
            framesT1 += 1
            owner = 1
        elif min(distances["T1"]) < min(distances["T2"]):
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
