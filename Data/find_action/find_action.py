import json

#home_player_ids = {2547, 2548, 2405, 201609, 204020, 2736, 2365, 101123, 2757, 1626159, 202355, 2617}
#visitor_player_ids = {101109, 101111, 200826, 203939, 101114, 201580, 203098, 202379, 202083, 202718, 2585, 2734, 1717}

visitor_player_ids = {202338, 204038, 101181, 201167, 204001, 2546, 201577, 203124, 1626170, 2756, 202682, 202498}
home_player_ids = {201166, 203926, 1626171, 2550, 2200, 203503, 202710, 201959, 203946, 202703, 202734, 201565, 201149}

basket_height = 10
def load_match_data(file_path):
    """Load match data from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def save_match_data(file_path, data):
    """Save match data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file)

def parse_state(state):
    """Parse state into meaningful components."""
    players_home = [(state[i], state[i+1]) for i in range(0, 10, 2)]  # Home team players' positions
    players_away = [(state[i], state[i+1]) for i in range(10, 20, 2)]  # Away team players' positions
    ball_position = (state[20], state[21], state[22])  # Ball position (x, y, z)
    game_clock = state[23] 
    shot_clock = state[24]  
    ball_owner = state[25]
    period = state[26]
    return players_home, players_away, ball_position, game_clock, shot_clock, ball_owner, period

def detect_action(state1, state2):
    """Detect the action that occurred between two states."""
    players_home1, players_away1, ball_position1, game_clock1, shot_clock1, ball_owner1, period1 = parse_state(state1)
    players_home2, players_away2, ball_position2, game_clock2, shot_clock2, ball_owner2, period2 = parse_state(state2)

    home_basket_coords = (88.8, 24.7)  # Ev sahibi takım sepet koordinatları (örnek)
    visitor_basket_coords = (5.37, 24.7)  # Rakip takım sepet koordinatları (örnek)

    actions = []  # actions = ["pass", "shoot", "dribble", "defend", "idle"]
    reward = 0

    if ball_owner1 in visitor_player_ids:  # Top rakipte
        if ball_owner2 in visitor_player_ids:
            actions.append("defend_fail")
            reward = 0
        elif ball_owner2 in home_player_ids:
            actions.append("succesfull_defend")
            reward = 0.2
        else:
            actions.append("idle_defend")
            reward = 0

    elif ball_owner1 in home_player_ids:  # Top ev sahibinde
        if ball_owner1 == ball_owner2:  # Dribble
            actions.append("dribble")
            reward = 0
        else:  # Pass or shot
            motion_vector = (ball_position2[0] - ball_position1[0], ball_position2[1] - ball_position1[1])
            basket_vector = (home_basket_coords[0] - ball_position2[0], home_basket_coords[1] - ball_position2[1])
            dot_product = motion_vector[0] * basket_vector[0] + motion_vector[1] * basket_vector[1]

            if dot_product > 0 and ball_position2[2] > 6.5:  # Shoot
                if (abs(ball_position2[0] - home_basket_coords[0]) < 2.0 and
                    abs(ball_position2[1] - home_basket_coords[1]) < 2.0 and
                    ball_position2[2] >= basket_height - 1.0): 
                    
                    actions.append("succesfull_shot")
                    reward = 1
                else:
                    actions.append("missed_shot")
                    reward = 0
                
            else:  # Pass
                if ball_owner2 in home_player_ids:
                    actions.append("succesfull_pass")
                    reward = 0.1
                elif ball_owner2 in visitor_player_ids:
                    actions.append("missed_pass")
                    reward = -0.2
                else:
                    actions.append("idle_pass")
                    reward = 0

    else:  # Sahipsiz top
        actions.append("idle")
        reward = 0

    return actions,reward

def update_json_with_actions(file_path):
    """Update the JSON file with actions and print them to console."""
    match_data = load_match_data(file_path)

    for i in range(len(match_data) - 1):
        state1 = match_data[i][0]
        state2 = match_data[i+1][0]
        actions, reward = detect_action(state1, state2)
        
        # Add actions to the JSON data
        match_data[i][1] = actions
        match_data[i][2] = reward

    # Save updated match data to the file
    save_match_data(output_file, match_data)
    print(f"Updated JSON file: {output_file}")

if __name__ == "__main__":
    input_file = "../filtered_data/fdni0021500493.json"  # Girdi JSON dosyasının adı
    output_file = "match_data_493.json"  # Çıktı JSON dosyasının adı

    update_json_with_actions(input_file)
