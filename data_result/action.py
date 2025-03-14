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
    home_score = csv_row[2]
    visitor_score = csv_row[3]
    
    for json_entry in json_data:

        if int(json_entry[0][0]) == period and int(json_entry[0][1]) == remaining_time: # TODO: 1 sanıye ıcınde ıkı tane shot yazdırabılıyoruz ektradan 6 tane fazla yazdırdık. bunun sebebı ise yuvarlama yapmak.
            print(period, remaining_time)
            json_entry[1] = "shot"
            break


# Değiştirilmiş JSON verisini kaydet
output_json_path = Path("action491.json")  # Çıktı JSON dosyasının tam yolunu buraya yazın
with open(output_json_path, 'w') as jsonfile:
    json.dump(json_data, jsonfile, indent=4)

print(f"JSON verisi güncellendi ve {output_json_path} dosyasına kaydedildi.")