import json
import numpy as np

# JSON dosyasını yükle
input_file = "../data/0021500493.json"  # Girdi JSON dosyasının adı
output_file = "fdni0021500493.json"  # Çıktı JSON dosyasının adı

# JSON verilerini yükle
with open(input_file, "r") as file:
    data = json.load(file)

# Filtrelenmiş veriler için yeni liste oluştur
filtered_events = []

for event in data["events"]:
    filtered_moments = []

    for i, moment in enumerate(event["moments"]):
        # Her 5. momenti ekle
        if i % 5 == 0:
            filtered_moments.append(moment)

    # Event içine filtrelenmiş momentleri ekle
    event["moments"] = filtered_moments
    filtered_events.append(event)

# Yeni veriyi kaydet
data["events"] = filtered_events

def identify_ball_holder(current_moment):
    ball_position = current_moment[5][0][2:4]
    player = np.array([[player[1], player[2], player[3]] for player in current_moment[5]])
    player = np.delete(player, 0, axis=0)  # Topu çıkar
    min_distance = 90
    min_id = 0
    for pos in player:
        player_position = pos[1:]
        distance = np.linalg.norm(player_position - ball_position) # euclidian 
        #print(distance)
        if (distance < min_distance):
            min_distance = distance
            min_id = pos[0]
            #print( min_distance)
            #print(min_id)
    min_id = int(min_id)
    if (min_distance < 2):
        return min_id
    else:
        return 0
    
match_data = []
for event in data["events"]:
    for moment in event["moments"]:
        #print(moment)
        player_positions = np.array([[player[2], player[3]] for player in moment[5]])
        player_positions = np.delete(player_positions, 0, axis=0)  # Topu çıkar
        ball_position = moment[5][0][2:]  # Topun koordinatları
        time_remaining = moment[3]  # Zaman bilgisi
        ball_holder = identify_ball_holder(moment)
        state = np.concatenate((player_positions.flatten(), ball_position, [time_remaining, ball_holder])) # TODO: state'lerimize skor eklenmeli 
        action = ["idle"]  # Örnek action
        reward = 0
        match_data.append([state.tolist(), action, reward])


with open(output_file, "w") as file:
    json.dump(match_data, file)

#print(f"Filtrelenmiş veri '{output_file}' dosyasına kaydedildi.")