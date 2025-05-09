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
        # --- 1) Pull out players & ball ---
        if isinstance(entry, dict):
            # original dict format
            players = entry.get("players")
            ball = entry.get("ball")
        elif isinstance(entry, tuple) and len(entry) == 3:
            # pipeline tuple format
            _, p, b = entry
            players, ball = p, b
        else:
            raise ValueError(f"Entry {i} must be dict or 3-tuple, got: {entry!r}")

        # --- 2) Normalize empties ---
        if not players:      # covers None or []
            players = None
        if isinstance(ball, list) and len(ball) == 0:
            ball = None

        # --- 3) Skip if no players, or first frame & no ball ---
        if players is None or (i == 0 and ball is None):
            cumulative_possessions.append(
                GetPossessionPercentage(frames, framesT1, framesT2)
            )
            team_possession_list.append(None)
            continue

        # --- 4) Handle multiple‐ball detections ---
        prev_pos = prevBall["field_position"] if prevBall else None
        if isinstance(ball, list) and len(ball) > 1:
            # if we have a valid prevBall in bounds, pick closest; else take first
            if prevBall and yardTL[0] <= prev_pos[0] <= yardTR[0] and yardBL[1] <= prev_pos[1] <= yardTL[1]:
                best = ball[0]
                best_d = DistanceBetweenObjects(best["field_position"], prev_pos)
                for det in ball[1:]:
                    pos = det["field_position"]
                    if yardTL[0] <= pos[0] <= yardTR[0] and yardBL[1] <= pos[1] <= yardTL[1]:
                        d = DistanceBetweenObjects(pos, prev_pos)
                        if d < best_d:
                            best_d, best = d, det
                ball = best
            else:
                ball = ball[0]

        # --- 5) If still no ball, maybe reuse prevBall ---
        if ball is None and i > 0:
            if prevBall and yardTL[0] <= prev_pos[0] <= yardTR[0] and yardBL[1] <= prev_pos[1] <= yardTL[1]:
                ball = prevBall
            else:
                cumulative_possessions.append(
                    GetPossessionPercentage(frames, framesT1, framesT2)
                )
                team_possession_list.append(None)
                continue

        # --- 6) Compute distances of each team to the ball ---
        distances = {"T1": [], "T2": []}
        for p in players:
            d = DistanceBetweenObjects(p["field_position"], ball["field_position"])
            if p["class_id"] == 1:
                distances["T1"].append(d)
            elif p["class_id"] == 2:
                distances["T2"].append(d)

        prevBall = ball  # stash for next frame

        # --- 7) If neither team detected near the ball, skip update ---
        if not distances["T1"] and not distances["T2"]:
            cumulative_possessions.append(
                GetPossessionPercentage(frames, framesT1, framesT2)
            )
            team_possession_list.append(None)
            continue

        # --- 8) Decide possession based on which team is closest (or only one present) ---
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
