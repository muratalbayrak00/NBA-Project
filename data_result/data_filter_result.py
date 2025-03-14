import json
import numpy as np

# JSON dosyasını yükle
input_file = "21500491_result.json"  # Girdi JSON dosyasının adı
output_file = "fdni21500491_result.json"  # Çıktı JSON dosyasının adı

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
        #print(moment)
        period = moment["period"]
        game_clock = moment["game_clock"]
        shot_clock = moment["shot_clock"]
        ball_position = moment["ball"][2:]
        player_positions = np.array([[player[2], player[3]] for player in moment["players"]])
        if (moment["ball_owner"] is None ):
            ball_owner = 0
        else:
            ball_owner = moment["ball_owner"]["player_id"]
            
        state = np.concatenate(([period, game_clock, shot_clock],ball_position, player_positions.flatten(), [ball_owner])) # TODO: state'lerimize skor eklenmeli
        if len(state) == 27:
            momentLength = len(event["moments"])-1
            if(event["end_action"] == "shot") and  i == momentLength:
                action = "shot"
            else:
                action = moment["action"]

            reward = 0
            match_data.append([state.tolist(), action, reward])
             

with open(output_file, "w") as file:
    json.dump(match_data, file)

print(f"Filtrelenmiş veri '{output_file}' dosyasına kaydedildi.")
