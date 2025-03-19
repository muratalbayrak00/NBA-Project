import csv
import json
from pathlib import Path

# CSV dosyasının yolu
csv_path = Path("../Data/fixed_shots/score_result.csv")  # CSV dosyanızın tam yolunu buraya yazın

# JSON dosyasının yolu
json_path = Path("fdni21500491_result.json")  # JSON dosyanızın tam yolunu buraya yazın

# CSV verisini oku
csv_data = []
with open(csv_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        csv_data.append([int(row[0]), int(row[1]), int(row[2]), int(row[3])])

# JSON verisini yükle
with open(json_path, 'r') as jsonfile:
    json_data = json.load(jsonfile)

# CSV verisindeki her bir satır için JSON verisinde eşleşen periyot ve kalan süre değerlerini bul ve action'ı "shot" olarak değiştir
for csv_row in csv_data:
    period = csv_row[0]
    remaining_time = csv_row[1]

    for json_entry in json_data:
        if int(json_entry[0][0]) == period and int(json_entry[0][1]) == remaining_time: # TODO: 1 sanıye ıcınde ıkı tane shot yazdırabılıyoruz ektradan 6 tane fazla yazdırdık. bunun sebebı ise yuvarlama yapmak.  
                json_entry[1] = "shot"
                break

first_shot_time = None
second_shot_time = 0

for json_entry in json_data:
    if json_entry[1] == "shot":
        first_shot_time = json_entry[0][1]  #525.08 , 524.96, 450
        diff = abs(second_shot_time - first_shot_time) #525 , 1 

        if diff < 5: 
            json_entry[1] = "" 

        second_shot_time = first_shot_time 

shot_data_csv = []
home_score = 0
away_score = 0
for csv_row in csv_data:
    period = csv_row[0]
    remaining_time = csv_row[1]
    is_home_team = csv_row[2]
    point = csv_row[3]
    #shot_data_csv.append([period,remaining_time,is_home_team,point])

    csv_index = 0

    for json_entry in json_data:
        game_clock = json_entry[0][1] 
        diff_time = abs(remaining_time - game_clock)

        if json_entry[1] == "shot" and diff_time < 5 and period == json_entry[0][0]:
            if is_home_team == 1:
                home_score = home_score + point
                json_entry[0][27] = home_score
                json_entry[0][28] = away_score
            elif is_home_team == 0:
                away_score = away_score + point
                json_entry[0][27] = home_score
                json_entry[0][28] = away_score

for json_entry in json_data:

    if json_entry[1] == "shot" and json_entry[0][27] == 0 and json_entry[0][28] == 0:
        json_entry[1] = ""

home_score = 0
away_score = 0
for json_entry in json_data:

    if json_entry[1] == "shot":
        home_score = json_entry[0][27]
        away_score = json_entry[0][28]
    
    json_entry[0][27] = home_score
    json_entry[0][28] = away_score

# Değiştirilmiş JSON verisini kaydet
output_json_path = Path("action491.json")  # Çıktı JSON dosyasının tam yolunu buraya yazın
with open(output_json_path, 'w') as jsonfile:
    json.dump(json_data, jsonfile)

print(f"JSON verisi güncellendi ve {output_json_path} dosyasına kaydedildi.")