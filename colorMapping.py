import math
teams_jersey_colors = {
  "FC Barcelona": [[0, 0, 139], [178, 34, 34], [255, 215, 0]],
  "Real Madrid": [[255, 255, 255], [0, 0, 139], [128, 128, 128]],
  "Manchester United": [[220, 20, 60], [255, 255, 255], [0, 0, 0]],
  "Manchester City": [[135, 206, 235], [0, 0, 128], [0, 0, 0]],
  "Liverpool": [[200, 16, 46], [255, 255, 255], [0, 100, 0]],
  "Chelsea": [[0, 0, 205], [255, 255, 0], [255, 255, 255]],
  "Arsenal": [[255, 0, 0], [255, 255, 255], [255, 215, 0], [0, 0, 0]],
  "Tottenham Hotspur": [[255, 255, 255], [0, 0, 128], [135, 206, 235]],
  "Bayern Munich": [[220, 20, 60], [255, 255, 255], [0, 0, 0]],
  "Borussia Dortmund": [[255, 215, 0], [0, 0, 0], [255, 69, 0]],
  "PSG": [[0, 0, 139], [220, 20, 60], [255, 192, 203]],
  "Juventus": [[255, 255, 255], [0, 0, 0], [0, 0, 255]],
  "AC Milan": [[139, 0, 0], [0, 0, 0], [255, 255, 255]],
  "Inter Milan": [[0, 0, 205], [0, 0, 0], [255, 255, 255]],
  "AS Roma": [[128, 0, 0], [255, 255, 0], [255, 255, 255]],
  "Napoli": [[135, 206, 235], [0, 0, 139], [0, 0, 0]],
  "Atletico Madrid": [[255, 0, 0], [255, 255, 255], [0, 0, 139]],
  "Sevilla": [[255, 255, 255], [220, 20, 60], [0, 0, 0]],
  "Ajax": [[255, 255, 255], [220, 20, 60], [0, 0, 0]],
  "FC Porto": [[0, 0, 205], [255, 255, 255], [255, 215, 0]],
  "Benfica": [[255, 0, 0], [255, 255, 255], [0, 0, 0]],
  "Sporting CP": [[0, 128, 0], [255, 255, 255], [0, 0, 0]],
  "Galatasaray": [[220, 20, 60], [255, 215, 0], [0, 0, 0]],
  "Fenerbahçe": [[255, 255, 0], [0, 0, 139], [255, 255, 255]],
  "Celtic": [[0, 128, 0], [255, 255, 255], [255, 255, 0]],
  "Rangers": [[0, 0, 205], [255, 255, 255], [255, 0, 0]],
  "Lyon": [[255, 255, 255], [0, 0, 139], [255, 0, 0]],
  "Marseille": [[255, 255, 255], [135, 206, 250], [0, 0, 128]],
  "Bayer Leverkusen": [[220, 20, 60], [0, 0, 0], [192, 192, 192]],
  "RB Leipzig": [[255, 255, 255], [255, 0, 0], [0, 0, 0]],
  "Real Betis": [[0, 128, 0], [255, 255, 255], [0, 100, 0]],
  "Elche CF": [[255, 255, 255], [0, 128, 0], [0, 0, 0]]
}

def color_distance(c1, c2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

def assign_best_unique_colors(color1, color2, team1, team2, teams_colors=teams_jersey_colors):
    if team1 not in teams_colors or team2 not in teams_colors:
        raise ValueError("One or both teams not found in jersey color dictionary.")

    combinations = []
    for color, label in zip([color1, color2], ['color1', 'color2']):
        for team in [team1, team2]:
            for jersey_color in teams_colors[team]:
                combinations.append({
                    'color_label': label,
                    'team': team,
                    'jersey_color': jersey_color,
                    'input_color': color,
                    'distance': color_distance(color, jersey_color)
                })

    best_pair = None
    best_total_distance = float('inf')

    for c1 in combinations:
        for c2 in combinations:
            if c1['color_label'] == c2['color_label']:
                continue
            if c1['team'] == c2['team']:
                continue

            total_dist = c1['distance'] + c2['distance']
            if total_dist < best_total_distance:
                print("**********\nCOLOR 1\n*********")
                print(f"color 1 = {c1['input_color']} in distance = {c1['distance']} from actual = {c1['jersey_color']}")
                print("**********\nCOLOR 2\n*********")
                print(f"and color 2 = {c2['input_color']} in distance = {c2['distance']} from actual = {c2['jersey_color']}")
                best_total_distance = total_dist
                best_pair = {
                    c1['team']: {'input_color': c1['input_color'],
                                 'jersey_color': c1['jersey_color']},
                    c2['team']: {'input_color': c2['input_color'],
                                 'jersey_color': c2['jersey_color']}
                }
            
    print("**********\nFinally\n*********")
    print(f"best_pair = {best_pair}")

    return best_pair