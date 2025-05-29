import json

# JSON verisini yükle
with open("21500491_last_result.json", "r") as f: # GSW için dosya adı
    data = json.load(f)

total_reward = 0
successfull_atack = 0
unsuccessfull_atack = 0
# Başlangıç değerleri
previous_owner = data[0][0][26]
previous_home_score = data[0][0][27]

for i in range(1, len(data)):
    current_period = data[i][0][0]
    if current_period == 4:

        current_owner = data[i][0][26]

        if previous_owner == 2 and current_owner == 1:
            previous_home_score = data[i][0][27]

        if previous_owner == 1 and current_owner == 2:
            current_home_score = data[i][0][27]
            # Top ev sahibinden rakibe geçti
            if current_home_score > previous_home_score:
                total_reward += 1
                successfull_atack += 1
            else:
                total_reward -= 1
                unsuccessfull_atack += 1
            previous_home_score = current_home_score

print("Başarılı atak sayısı:", successfull_atack)
print("Başarısız atak sayısı:", unsuccessfull_atack)     

print("İlk periyot için ev sahibi takımın toplam ödülü:", total_reward)
