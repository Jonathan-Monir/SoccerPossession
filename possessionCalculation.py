def CalculatePossession(data, yardTL, yardTR, yardBL, yardBR):
    frames = 0
    framesT1 = framesT2 = 0
    cumulative_possessions = []
    team_possession_list = []
    prevBall = None

    for i, entry in enumerate(data):
        # --- DEBUG: inspect first few entries ---
        if i < 3:
            print(f"\n--- entry #{i} ---")
            print("type(entry):", type(entry))
            # attempt dict access
            try:
                print("entry['players']:", entry['players'])
                print("entry['ball']   :", entry['ball'])
            except Exception as e:
                print("  not dict:", repr(entry))

        # --- 1) Extract players & ball ---
        if isinstance(entry, dict):
            players = entry.get('players')
            ball     = entry.get('ball')
        elif isinstance(entry, tuple):
            # adjust this unpacking once we know the tuple shape
            # for now, stash raw tuple in a,b,c
            # you’ll see in debug which position is which
            a, b, c = entry
            # placeholder—overwrite below once you know:
            players, ball = b, c
        else:
            raise ValueError(f"Entry {i} must be dict or tuple, got {type(entry)}")

        # --- 2) Normalize empties ---
        if not players:       # covers None or []
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

        # --- 4) Handle multiple ball detections ---
        prev_pos = prevBall['field_position'] if prevBall else None
        if isinstance(ball, list) and len(ball) > 1:
            if (
                prevBall
                and yardTL[0] <= prev_pos[0] <= yardTR[0]
                and yardBL[1] <= prev_pos[1] <= yardTL[1]
            ):
                best = ball[0]
                best_d = DistanceBetweenObjects(best['field_position'], prev_pos)
                for det in ball[1:]:
                    pos = det['field_position']
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

        # --- 5) Reuse previous ball if missing ---
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

        # --- 6) Compute distances ---
        distances = {'T1': [], 'T2': []}
        for p in players:
            d = DistanceBetweenObjects(
                p['field_position'], ball['field_position']
            )
            if p['class_id'] == 1:
                distances['T1'].append(d)
            elif p['class_id'] == 2:
                distances['T2'].append(d)

        prevBall = ball

        # --- 7) Skip if no detections near ball ---
        if not distances['T1'] and not distances['T2']:
            cumulative_possessions.append(
                GetPossessionPercentage(frames, framesT1, framesT2)
            )
            team_possession_list.append(None)
            continue

        # --- 8) Decide and record possession ---
        if not distances['T1']:
            framesT2 += 1
            owner = 2
        elif not distances['T2']:
            framesT1 += 1
            owner = 1
        elif min(distances['T1']) < min(distances['T2']):
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
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5


def GetPossessionPercentage(frames, f1, f2):
    if frames == 0:
        return {'possT1': 0, 'possT2': 0}
    return {
        'possT1': (f1 / frames) * 100,
        'possT2': (f2 / frames) * 100
    }
