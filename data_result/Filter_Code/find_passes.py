import json
import Filter_Code.action as action_file

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
        
def get_player_id(game_id):

    file_path = f"Row_data/{game_id}_result.json"

    with open(file_path, "r") as file:
        data = json.load(file)

    home_player_ids = set()
    visitor_player_ids = set()

    for event in data:
        for moment in event["moments"]:
            players = moment["players"]

            # İlk oyuncunun team_id'sini alarak home/visitor ayrımını yap
            home_team_id = players[0][0]
            visitor_team_id = None
            
            # Visitor team_id'yi belirle
            for player in players:
                if player[0] != home_team_id:
                    visitor_team_id = player[0]
                    break
            
            # Oyuncu ID'lerini sınıflandır
            for player in players:
                team_id, player_id = player[0], player[1]
                if team_id == home_team_id:
                    home_player_ids.add(player_id)
                elif team_id == visitor_team_id:
                    visitor_player_ids.add(player_id)

    return list(home_player_ids), list(visitor_player_ids)

def update_json_with_actions(game_id):
    home_player_ids, visitor_player_ids = get_player_id(game_id)

    input_json = action_file.find_action(game_id)
    #input_file = "action491.json"  # Girdi JSON dosyasının adı
    #output_file = "with_passes491.json"  # Çıktı JSON dosyasının adı
    output_file = f"Last_result_data/{game_id}_last_result.json"

    """Update the JSON file with actions and print them to console."""
    match_data = input_json
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
        # shot lara reward verme islemini action.py da score islemini yazarken yapiyoruz
        
        match_data[i][1] = action
        match_data[i][2] = reward
    # bu fonksiyon baslari buluyor
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

    # home player id lere 1, visitor id lere 2 ataniyor burda
    for i in range(len(match_data) - 1):
        state = match_data[i][0]
        if state[26] in home_player_ids:
            state[26] = 1
        elif state[26] in visitor_player_ids:
            state[26] = 2       

    save_match_data(output_file, match_data)
    print(f"Updated JSON file: {output_file}")

if __name__ == "__main__":
    game_id = "21500491"
    update_json_with_actions(game_id)
