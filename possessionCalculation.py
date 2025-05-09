def CalculatePossession(data,
                        yardTL=[0.0, 68.0],
                        yardTR=[105.0, 68.0],
                        yardBL=[0.0, 0.0],
                        yardBR=[105.0, 0.0]):
    frames = 0
    framesT1 = framesT2 = 0
    cumulative_possessions = []
    team_possession_list = []
    prevBall = None

    for i, (frame, players, ball) in enumerate(data):
        distances = {"T1": [], "T2": []}

        # If no players or (first frame & no ball)
        if players is None or (i == 0 and ball is None):
            cumulative_possessions.append(
                GetPossessionPercentage(frames, framesT1, framesT2)
            )
            team_possession_list.append(None)
            continue

        prevBall_position = prevBall["field_position"] if prevBall else None

        # Handle multiple ball detections
        if isinstance(ball, list) and len(ball) > 1:
            # If prevBall is valid & inside yard bounds, pick the closest detection
            if (
                prevBall
                and yardTL[0] <= prevBall_position[0] <= yardTR[0]
                and yardBL[1] <= prevBall_position[1] <= yardTL[1]
            ):
                currBall = ball[0]
                minDist = DistanceBetweenObjects(currBall["field_position"], prevBall_position)
                for b in ball[1:]:
                    pos = b["field_position"]
                    if (
                        yardTL[0] <= pos[0] <= yardTR[0]
                        and yardBL[1] <= pos[1] <= yardTL[1]
                    ):
                        d = DistanceBetweenObjects(pos, prevBall_position)
                        if d < minDist:
                            minDist, currBall = d, b
                ball = currBall
            else:
                ball = ball[0]

        # If ball missing, reuse prevBall if it's valid
        if i > 0 and ball is None:
            if (
                prevBall
                and yardTL[0] <= prevBall_position[0] <= yardTR[0]
                and yardBL[1] <= prevBall_position[1] <= yardTL[1]
            ):
                ball = prevBall
            else:
                cumulative_possessions.append(
                    GetPossessionPercentage(frames, framesT1, framesT2)
                )
                team_possession_list.append(None)
                continue

        # Compute distances from each player to the ball
        for p in players:
            dist = DistanceBetweenObjects(p["field_position"], ball["field_position"])
            if p["class_id"] == 1:
                distances["T1"].append(dist)
            elif p["class_id"] == 2:
                distances["T2"].append(dist)

        prevBall = ball  # save for next frame

        # If neither team has any players detected near the ball
        if not distances["T1"] and not distances["T2"]:
            cumulative_possessions.append(
                GetPossessionPercentage(frames, framesT1, framesT2)
            )
            team_possession_list.append(None)
            continue

        # Decide who possesses based on min distance (or only one team present)
        if not distances["T1"]:
            framesT2 += 1
            team_id = 2
        elif not distances["T2"]:
            framesT1 += 1
            team_id = 1
        elif min(distances["T1"]) < min(distances["T2"]):
            framesT1 += 1
            team_id = 1
        else:
            framesT2 += 1
            team_id = 2

        frames += 1
        team_possession_list.append(team_id)
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
