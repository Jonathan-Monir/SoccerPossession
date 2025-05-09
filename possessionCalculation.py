def CalculatePossession(data, yardTL=[0.0, 68.0], yardTR=[105.0, 68.0], yardBL=[0.0, 0.0], yardBR=[105.0, 0.0]):
    frames = 0
    framesT1 = framesT2 = 0
    cumulative_possessions = []
    team_possession_list = []
    prevBall = None

    for i, (frame_index, ball, players) in enumerate(data):
        distances = { "T1": [], "T2": [] }

        if players is None or (i == 0 and ball is None):
            cumulative_possessions.append(GetPossessionPercentage(frames, framesT1, framesT2))
            team_possession_list.append(None)
            continue

        prevBall_position = prevBall["field_position"] if prevBall is not None else None

        # Process ball when it's a list
        if isinstance(ball, list) and len(ball) > 1:
            if prevBall is not None and prevBall_position[0] >= yardTL[0] and prevBall_position[0] >= yardTR[0] and yardBL[1] >= prevBall_position[1] >= yardTL[1]:
                currBall = ball[0]
                minDist = DistanceBetweenObjects(currBall["field_position"], prevBall_position)
                for idx, b in enumerate(ball):
                    if idx == 0:
                        continue
                    b_position = b["field_position"]
                    if b_position[0] >= yardTL[0] and b_position[0] >= yardTR[0] and yardBL[1] >= b_position[1] >= yardTL[1]:
                        d = DistanceBetweenObjects(b_position, prevBall_position)
                        if d < minDist:
                            minDist = d
                            currBall = b
                ball = currBall
            else:
                ball = ball[0]

        # Use previous ball if current ball is missing
        if i > 0 and ball is None:
            if prevBall is not None and prevBall_position[0] >= yardTL[0] and prevBall_position[0] >= yardTR[0] and yardBL[1] >= prevBall_position[1] >= yardTL[1]:
                ball = prevBall
            else:
                cumulative_possessions.append(GetPossessionPercentage(frames, framesT1, framesT2))
                team_possession_list.append(None)
                continue

        # Calculate distances for each team
        for player in players:
            if player["class_id"] == 1:
                distances["T1"].append(DistanceBetweenObjects(player["field_position"], ball["field_position"]))
            elif player["class_id"] == 2:
                distances["T2"].append(DistanceBetweenObjects(player["field_position"], ball["field_position"]))

        prevBall = ball

        distancesT1 = len(distances["T1"])
        distancesT2 = len(distances["T2"])

        if distancesT1 == 0 and distancesT2 == 0:
            cumulative_possessions.append(GetPossessionPercentage(frames, framesT1, framesT2))
            team_possession_list.append(None)
            continue

        # Determine possession
        if distancesT1 == 0:
            framesT2 += 1
            team_possession = 2
        elif distancesT2 == 0:
            framesT1 += 1
            team_possession = 1
        elif min(distances["T1"]) < min(distances["T2"]):
            framesT1 += 1
            team_possession = 1
        else:
            framesT2 += 1
            team_possession = 2

        frames += 1
        team_possession_list.append(team_possession)
        cumulative_possessions.append(GetPossessionPercentage(frames, framesT1, framesT2))

    return cumulative_possessions, team_possession_list


def DistanceBetweenObjects(player, ball):
    return ((player[0] - ball[0])**2 + (player[1] - ball[1])**2)**0.5


def GetPossessionPercentage(frames, framesT1, framesT2):
    if frames == 0:
        return {"possT1": 0, "possT2": 0}
    possT1 = (framesT1 / frames) * 100
    possT2 = (framesT2 / frames) * 100
    return {"possT1": possT1, "possT2": possT2}
