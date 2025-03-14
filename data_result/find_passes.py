import json

home_player_ids = {2547, 2548, 2405, 201609, 204020, 2736, 2365, 101123, 2757, 1626159, 202355, 2617}
visitor_player_ids = {101109, 101111, 200826, 203939, 101114, 201580, 203098, 202379, 202083, 202718, 2585, 2734, 1717}

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

    period = state[0]
    game_clock = state[1]
    shot_clock = state[2]
    ball_position = (state[3], state[4], state[5])  # Ball position (x, y, z)
    players_home = [(state[i], state[i+1]) for i in range(6, 16, 2)]  # Home team players' positions
    players_away = [(state[i], state[i+1]) for i in range(16, 26, 2)]  # Away team players' positions
    ball_owner = state[26]

    
    return period, game_clock, shot_clock ,ball_position, players_home, players_away, ball_owner

def detect_action(state1, state2, actions, reward):

    """Detect the action that occurred between two states."""

    period1, game_clock1, shot_clock1 ,ball_position1, players_home1, players_away1, ball_owner1 = parse_state(state1)
    period2, game_clock2, shot_clock2 ,ball_position2, players_home2, players_away2, ball_owner2 = parse_state(state2)
    if actions == "miss":
        actions = ""

    if actions == "shot":
        reward = 1
    elif actions == "pass":
        reward = 0.1
    elif ball_owner1 in home_player_ids :
        if ball_owner2 in home_player_ids and ball_owner1 != ball_owner2:
            actions = "pass"
            reward = 0.1
        elif ball_owner2 in visitor_player_ids:
            actions = "pass"
            reward = -0.2       

    return actions,reward



def update_json_with_actions(file_path):
    """Update the JSON file with actions and print them to console."""
    match_data = load_match_data(file_path)

    for i in range(len(match_data) - 1):
        state1 = match_data[i][0]
        state2 = match_data[i+1][0]
        actions = match_data[i][1]
        reward = match_data[i][2] 

        actions, reward = detect_action(state1, state2, actions, reward)
        

        # Add actions to the JSON data
        match_data[i][1] = actions
        match_data[i][2] = reward
            
    # Save updated match data to the file
    save_match_data(output_file, match_data)
    print(f"Updated JSON file: {output_file}")




if __name__ == "__main__":
    input_file = "action491.json"  # Girdi JSON dosyasının adı
    output_file = "with_passes491.json"  # Çıktı JSON dosyasının adı
    update_json_with_actions(input_file)
