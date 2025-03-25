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
    #with open(output_file, 'w', encoding='utf-8') as f:
        #json.dump(data, f, indent=4, ensure_ascii=False)  # Girintili JSON formatı
        
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
    home_pass = 0
    visitor_pass = 0
    for i in range(len(match_data) - 1):
        state = match_data[i][0]
        action = match_data[i][1]
        reward = match_data[i][2]

        if action == "pass":
            if state[26] in home_player_ids:
                reward = 0.1
                home_pass += 1
                
            elif state[26] in visitor_player_ids:
                action = ""
                visitor_pass += 1
                
        elif action == "shot":
            reward = 1
        match_data[i][1] = action
        match_data[i][2] = reward
    #print("home pass",home_pass)
    #print("visitor pass",visitor_pass)
    
    for i in range(len(match_data) - 1):
        state1 = match_data[i][0] 
        
        ball_owner1 = state1[26]  # İlk state'deki top sahibi
        ball_owner2 = None
        
        if ball_owner1 in home_player_ids:
            boolean= True
            # Maksimum 5 state boyunca pası kontrol et
            for j in range(1, 6):
                if i + j < len(match_data):
                    state2 = match_data[i + j][0]
                    ball_owner2 = state2[26]  # Sonraki state'deki geçici top sahibi
                    
                    if ball_owner2 != ball_owner1:  # top elden cikti
                        if boolean == True:
                            temp = j
                            boolean = False
                        if ball_owner2 != 0 and match_data[i+temp-1][1] != "shot":  # Top sahibi değişmiş mi?
                            if ball_owner2 in home_player_ids:
                                match_data[i+temp-1][1] = "pass" 
                                match_data[i+temp-1][2] = 0.1
                                #actions = "bbpass" # basarili
                            elif ball_owner2 in visitor_player_ids:
                                match_data[i+temp-1][1] = "pass" # basarisiz
                                match_data[i+temp-1][2] = -0.2
                            break  # Pası bulduk, daha fazla state kontrol etmeye gerek yok
                        
    for i in range(len(match_data) - 1):
        state = match_data[i][0]
        action = match_data[i][1]
        reward = match_data[i][2]
        if action != "shot" and action != "pass":
            if state[26] in visitor_player_ids:
                action = "defend"
                if match_data[i+1][0][26] in home_player_ids:
                    reward = 0.2
            if state[26] in home_player_ids:
                action = "dribble"
                if match_data[i+1][0][26] in visitor_player_ids:
                    reward = -0.2
            match_data[i][1] = action
            match_data[i][2] = reward

    save_match_data(output_file, match_data)
    print(f"Updated JSON file: {output_file}")

if __name__ == "__main__":
    input_file = "action491.json"  # Girdi JSON dosyasının adı
    output_file = "with_passes491.json"  # Çıktı JSON dosyasının adı
    update_json_with_actions(input_file)
