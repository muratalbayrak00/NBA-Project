import json
import numpy as np

def filter_data(game_id):
    # JSON dosyasını yükle
    input_file = f"Row_data/{game_id}_result.json"  # Girdi JSON dosyasının adı
    output_file = f"fdni{game_id}_result.json"  # Çıktı JSON dosyasının adı

    # JSON verilerini yükle
    with open(input_file, "r") as file:
        data = json.load(file)


    # Filtrelenmiş veriler için yeni liste oluştur
    filtered_events = []


    for event in data:
        filtered_moments = []

        for i, moment in enumerate(event["moments"]):
            # Her 5. momenti ekle
            if i % 5 == 0:
                filtered_moments.append(moment)

        # Event içine filtrelenmiş momentleri ekle
        event["moments"] = filtered_moments
        filtered_events.append(event)

    # Yeni veriyi kaydet
    data = filtered_events

    match_data = []
    for event in data:
        for i,moment in enumerate(event["moments"]):
            period = moment["period"]
            game_clock = moment["game_clock"]
            shot_clock = moment["shot_clock"]
            ball_position_x = int(moment["ball"][2])
            ball_position_y = int(moment["ball"][3])
            ball_position_z = int(moment["ball"][4])
            player_positions = np.array([[int(player[2]), int(player[3])] for player in moment["players"]])
            if (moment["ball_owner"] is None ):
                ball_owner = 0
            else:
                ball_owner = moment["ball_owner"]["player_id"]

            home_score = 0
            visitor_score = 0
                
            state = np.concatenate(([period, game_clock, shot_clock],[ball_position_x, ball_position_y, ball_position_z], player_positions.flatten(), [ball_owner, home_score, visitor_score])) # TODO: state'lerimize skor eklenmeli
            if len(state) == 29:
                momentLength = len(event["moments"])-1
                if(event["end_action"] == "shot") and  i == momentLength:
                    action = "shot"
                if(event["start_action"] == "shot") and  i == 0:
                    action = "shot"
                else:
                    action = moment["action"]

                reward = 0
                match_data.append([state.tolist(), action, reward])
                

    #with open(output_file, "w") as file:
        #json.dump(match_data, file)

    print(f"Filtrelenmiş veri '{output_file}' dosyasına kaydedildi.")

    return match_data


if __name__ == "__main__":
    game_id = "0021500491"
    filter_data(game_id)