import json

# JSON verisini yükle
with open("21500005_last_result.json", "r") as f: # PHI için dosya adı   away!!
    data = json.load(f)

total_reward = 0
successfull_atack = 0
unsuccessfull_atack = 0
# Başlangıç değerleri
previous_owner = data[0][0][26]
previous_away_score = data[0][0][28]

for i in range(1, len(data)):
    current_period = data[i][0][0]
    if current_period == 1:

        current_owner = data[i][0][26]

        if previous_owner == 1 and current_owner == 2:
            previous_away_score = data[i][0][28]
            previous_owner = current_owner

        if previous_owner == 2 and current_owner == 1:
            current_away_score = data[i][0][28]
        
            if current_away_score > previous_away_score:
                total_reward += 1
                successfull_atack += 1
            else:
                total_reward -= 1
                unsuccessfull_atack += 1
            previous_away_score = current_away_score
            previous_owner = current_owner

print("Başarılı atak sayısı:", successfull_atack)
print("Başarısız atak sayısı:", unsuccessfull_atack)     

print("İlk periyot için ev sahibi takımın toplam ödülü:", total_reward)
