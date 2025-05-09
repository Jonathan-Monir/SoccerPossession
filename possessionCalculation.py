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
        # --- DEBUG: inspect entry, players & ball types/content ---
        if i < 5:
            print(f"\n=== FRAME {i} DEBUG ===")
            print(" raw entry :", repr(entry))
            # attempt dict access
            if isinstance(entry, dict):
                print(" entry is dict")
                pl = entry.get("players")
                ba = entry.get("ball")
            elif isinstance(entry, tuple):
                print(f" entry is tuple, len={len(entry)}")
                # show each element
                for idx, elem in enumerate(entry):
                    print(f"  elem[{idx}] ({type(elem)}): {repr(elem)}")
                # heuristically guess:
                # if second elem is list-of-dicts with class_id 1/2 → players
                # if second elem is dict/class_id 0 or list thereof → ball
                a, b, *rest = entry
                # you'll see below which is which
                pl, ba = None, None
                if isinstance(b, list) and b and isinstance(b[0], dict) and b[0].get("class_id") in (1,2):
                    pl = b
                    ba = entry[2] if len(entry) > 2 else None
                else:
                    ba = b
                    pl = entry[2] if len(entry) > 2 else None
            else:
                print(" entry is neither dict nor tuple, type =", type(entry))
                pl = ba = None

            print("  → players:", type(pl), repr(pl))
            print("  → ball   :", type(ba), repr(ba))

        # --- now actually assign players & ball for logic below ---
        if isinstance(entry, dict):
            players = entry.get("players")
            ball     = entry.get("ball")
        elif isinstance(entry, tuple) and len(entry) >= 3:
            # adjust this mapping once you see the debug above
            # for now, assume elem[1] = players, elem[2] = ball:
            players, ball = entry[1], entry[2]
        else:
            # fallback: try any attributes
            raise ValueError(f"Frame {i}: unexpected entry format: {type(entry)}")

        # --- normalize empties ---
        if not players:
            players = None
        if isinstance(ball, list) and len(ball) == 0:
            ball = None

        # --- skip frames with no data ---
        if players is None or (i == 0 and ball is None):
            cumulative_possessions.append(
                GetPossessionPercentage(frames, framesT1, framesT2)
            )
            team_possession_list.append(None)
            continue

        # --- handle multiple-ball detections ---
        prev_pos = prevBall["field_position"] if prevBall else None
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

        # --- reuse previous ball if missing ---
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

        # --- compute distances ---
        distances = {"T1": [], "T2": []}
        for p in players:
            # **DEBUG**: make sure ball is still dict or list here
            if not isinstance(ball, dict):
                print(f"ERROR: frame {i} — ball is not dict: {ball!r}")
                raise TypeError(f"Unexpected ball type at frame {i}: {type(ball)}")
            d = DistanceBetweenObjects(p["field_position"], ball["field_position"])
            if p["class_id"] == 1:
                distances["T1"].append(d)
            elif p["class_id"] == 2:
                distances["T2"].append(d)

        prevBall = ball

        # --- skip if neither team near ball ---
        if not distances["T1"] and not distances["T2"]:
            cumulative_possessions.append(
                GetPossessionPercentage(frames, framesT1, framesT2)
            )
            team_possession_list.append(None)
            continue

        # --- decide possession ---
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
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5


def GetPossessionPercentage(frames, f1, f2):
    if frames == 0:
        return {'possT1': 0, 'possT2': 0}
    return {
        'possT1': (f1 / frames) * 100,
        'possT2': (f2 / frames) * 100
    }
