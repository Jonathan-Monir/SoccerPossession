    for i, entry in enumerate(data):
        frame, ball, players = entry


        ball = ValidateBall(ball)
        players = ValidatePlayers(players)
        ball = HandleBallWithValidation(ball, prevBall, yardTL, yardTR, yardBL, yardTL[1])

        print(f"\nFrame {i}:")
        print(f"  Ball: {ball}")
        print(f"  Players: {players}")

        if ball and isinstance(ball, dict) and "field_position" in ball:
            owner = GetClosestTeam(ball["field_position"], players)
            print(f"  Closest owner team: {owner}")
            if owner == 1:
                framesT1 += 1
                frames += 1
            elif owner == 2:
                framesT2 += 1
                frames += 1
            team_possession_list.append(owner)
        else:
            print("  Skipped frame due to invalid ball or players")
            team_possession_list.append(None)

        cumulative_possessions.append(
            GetPossessionPercentage(frames, framesT1, framesT2)
        )
        prevBall = ball if isinstance(ball, dict) else None
