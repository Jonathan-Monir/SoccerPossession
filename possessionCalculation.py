def CalculatePossession(data, yardTL=[0.0, 68.0], yardTR=[105.0, 68.0], yardBL=[0.0, 0.0], yardBR=[105.0, 0.0]):
    frames = 0
    framesT1 = framesT2 = 0
    cumulative_possessions = []
    team_possession_list = []
    prevBall = None

    for i, frame in enumerate(data):
        distances = { "T1": [], "T2": [] }
        players = frame["players"]
        ball = frame["ball"]

        if players is None or (i == 0 and ball is None):
            cumulative_possessions.append(GetPossessionPercentage(frames, framesT1, framesT2))
            team_possession_list.append(None)  # No possession decision available
            continue

        prevBall_position = prevBall["field_position"] if prevBall is not None else None

        # Process ball when it's a list (multiple ball detections)
        if isinstance(ball, list) and len(ball) > 1:
            if prevBall is not None and prevBall_position[0] >= yardTL[0] and prevBall_position[0] >= yardTR[0] and prevBall_position[1] >= yardTL[1] and prevBall_position[1] <= yardBL[1]:
                currBall = ball[0]
                minDist = DistanceBetweenObjects(currBall["field_position"], prevBall_position)
                for idx, b in enumerate(ball):
                    if idx == 0:
                        continue
                    b_position = b["field_position"]
                    if b_position[0] >= yardTL[0] and b_position[0] >= yardTR[0] and b_position[1] >= yardTL[1] and b_position[1] <= yardBL[1]:
                        d = DistanceBetweenObjects(b_position, prevBall_position)
                        if d < minDist:
                            minDist = d
                            currBall = b
                    else:
                        ball.pop(idx)
                ball = currBall
            else:
                ball = ball[0]

        # Use previous ball if current ball is missing
        if i > 0 and frame["ball"] is None:
            if prevBall is not None and prevBall_position[0] >= yardTL[0] and prevBall_position[0] >= yardTR[0] and prevBall_position[1] >= yardTL[1] and prevBall_position[1] <= yardBL[1]:
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
        # If no distances calculated, append current cumulative percentage and skip possession update
        if distancesT1 == 0 and distancesT2 == 0:
            cumulative_possessions.append(GetPossessionPercentage(frames, framesT1, framesT2))
            team_possession_list.append(None)
            continue

        # Determine possession based on distances
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



# data = [
#     {
#         "frame_index": 0,
#         "players": [
#             {"class_id": 1, "field_position": [62.53, 43.14]},
#             {"class_id": 1, "field_position": [91.73, 35.23]},
#             {"class_id": 1, "field_position": [60.05, 36.33]},
#             {"class_id": 2, "field_position": [66.44, 51.01]},
#             {"class_id": 2, "field_position": [57.66, 40.67]},
#             {"class_id": 1, "field_position": [68.74, 23.63]},
#             {"class_id": 1, "field_position": [70.96, 41.67]},
#             {"class_id": 2, "field_position": [65.72, 16.64]},
#             {"class_id": 2, "field_position": [52.59, 38.76]},
#             {"class_id": 2, "field_position": [72.84, 36.52]}
#         ],
#         "ball": {"class_id": 0, "field_position": [62.53, 43.14]}
#     },
#     {
#         "frame_index": 1,
#         "players": [
#             {"class_id": 1, "field_position": [62.75, 43.50]},
#             {"class_id": 1, "field_position": [91.50, 35.10]},
#             {"class_id": 1, "field_position": [60.20, 36.50]},
#             {"class_id": 2, "field_position": [66.30, 51.20]},
#             {"class_id": 2, "field_position": [57.50, 40.80]},
#             {"class_id": 1, "field_position": [68.90, 23.80]},
#             {"class_id": 1, "field_position": [71.10, 41.90]},
#             {"class_id": 2, "field_position": [65.80, 16.80]},
#             {"class_id": 2, "field_position": [52.40, 38.50]},
#             {"class_id": 2, "field_position": [72.50, 36.90]}
#         ],
#         "ball": [
#             {"class_id": 0, "field_position": [78.10, 29.00]},
#             {"class_id": 0, "field_position": [56.00, 44.80]}
#         ]
#     },
#     {
#         "frame_index": 2,
#         "players": [
#             {"class_id": 1, "field_position": [63.00, 43.80]},
#             {"class_id": 1, "field_position": [91.20, 35.00]},
#             {"class_id": 1, "field_position": [60.40, 36.60]},
#             {"class_id": 2, "field_position": [66.10, 51.40]},
#             {"class_id": 2, "field_position": [57.30, 40.90]},
#             {"class_id": 1, "field_position": [69.10, 24.00]},
#             {"class_id": 1, "field_position": [71.30, 42.10]},
#             {"class_id": 2, "field_position": [65.90, 17.00]},
#             {"class_id": 2, "field_position": [52.20, 38.30]},
#             {"class_id": 2, "field_position": [72.20, 37.20]}
#         ],
#         "ball": [
#             {"class_id": 0, "field_position": [78.80, 29.50]},
#             {"class_id": 0, "field_position": [56.80, 44.20]}
#         ]
#     }
# ]
# data = None



# yardTL, yardTR, yardBL, yardBR = [29.0, 17.0], [45.5, 17.0], [29.0, 26.0], [45.5, 26.0]
# poss = CaculatePossession(data, yardTL, yardTR, yardBL, yardBR)
# print(poss)
