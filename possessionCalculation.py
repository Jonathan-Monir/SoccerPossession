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

    for i, tup in enumerate(data):
        # Expecting a 3-tuple: (frame, x, y)
        if not isinstance(tup, tuple) or len(tup) != 3:
            raise ValueError(f"Entry {i} is not a 3-tuple: {tup!r}")

        frame, a, b = tup

        # Decide which of a,b is ball vs players:
        # - ball is either a dict {"class_id":0,...} or list of such dicts
        # - players is list of dicts with class_id 1 or 2
        def looks_like_ball(obj):
            if isinstance(obj, dict) and obj.get("class_id") == 0:
                return True
            if isinstance(obj, list) and obj and isinstance(obj[0], dict) and obj[0].get("class_id") == 0:
                return True
            return False

        if looks_like_ball(a):
            ball, players = a, b
        else:
            players, ball = a, b

        # normalize empties
        if not players:
            players = None
        if isinstance(ball, list) and len(ball) == 0:
            ball = None

        # skip if no players or first frame with no ball
        if players is None or (i == 0 and ball is None):
            cumulative_possessions.append(
                GetPossessionPercentage(frames, framesT1, framesT2)
            )
            team_possession_list.append(None)
            continue

        prev_pos = prevBall["field_position"] if prevBall else None

        # if multiple ball detections, pick closest to previous (if valid), else first
        if isinstance(ball, list) and len(ball) > 1:
            if prevBall and yardTL[0] <= prev_pos[0] <= yardTR[0] and yardBL[1] <= prev_pos[1] <= yardTL[1]:
                curr, best_d = ball[0], DistanceBetweenObjects(ball[0]["field_position"], prev_pos)
                for det in ball[1:]:
                    pos = det["field_position"]
                    if yardTL[0] <= pos[0] <= yardTR[0] and yardBL[1] <= pos[1] <= yardTL[1]:
                        d = DistanceBetweenObjects(pos, prev_pos)
                        if d < best_d:
                            best_d, curr = d, det
                ball = curr
            else:
                ball = ball[0]

        # if ball still missing, reuse prevBall if valid
        if ball is None and i > 0:
            if prevBall and yardTL[0] <= prev_pos[0] <= yardTR[0] and yardBL[1] <= prev_pos[1] <= yardTL[1]:
                ball = prevBall
            else:
                cumulative_possessions.append(
                    GetPossessionPercentage(frames, framesT1, framesT2)
                )
                team_possession_list.append(None)
                continue

        # compute distances
        distances = {"T1": [], "T2": []}
        for p in players:
            d = DistanceBetweenObjects(p["field_position"], ball["field_position"])
            if p["class_id"] == 1:
                distances["T1"].append(d)
            elif p["class_id"] == 2:
                distances["T2"].append(d)

        prevBall = ball

        # if neither team near ball
        if not distances["T1"] and not distances["T2"]:
            cumulative_possessions.append(
                GetPossessionPercentage(frames, framesT1, framesT2)
            )
            team_possession_list.append(None)
            continue

        # assign possession
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
