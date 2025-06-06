import csv
import json
from pathlib import Path
import Filter_Code.data_filter as data_filter
import Filter_Code.find_score as find_score
import io

def find_action(game_id):
    # CSV dosyasının yolu
    #score_result = Path("../Data/fixed_shots/score_result.csv")  
    score_result = find_score.find_score_result(game_id) # score_result.csv den game_id e gore filtreleme yapip geri donderir
    
    # JSON dosyasının yolu
    #json_data = Path("fdni21500491_result.json") 
    json_data = data_filter.filter_data(game_id) # fdni li halini geri donderecek 

    # CSV verisini oku
    csv_data = []
    #with open(score_result, newline='') as csvfile:
        #reader = csv.reader(csvfile)
    #reader = csv.reader(score_result)
    reader = csv.reader(io.StringIO(score_result))

    for row in reader:
        csv_data.append([int(row[0]), int(row[1]), int(row[2]), int(row[3])])

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
            reward = json_entry[2]
            
            if json_entry[1] == "shot" and diff_time < 5 and period == json_entry[0][0]:
                if is_home_team == 1: # eger basarili atisi yapan takim ev sahibi ise 
                    home_score = home_score + point
                    json_entry[0][27] = home_score
                    json_entry[0][28] = away_score
                    reward = 1
                elif is_home_team == 0: # eger basarili atisi yapan takim rakip sahibi ise 
                    away_score = away_score + point
                    json_entry[0][27] = home_score
                    json_entry[0][28] = away_score
                    reward = -1
            json_entry[2] = reward

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
   # output_json_path = Path("data_result/action491.json")  # Çıktı JSON dosyasının tam yolunu buraya yazın
    #with open(output_json_path, 'w') as jsonfile:
    #    json.dump(json_data, jsonfile)

    return json_data
if __name__ == "__main__":
    game_id = "0021500491"
    find_action(game_id)