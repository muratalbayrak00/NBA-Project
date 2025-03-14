import pandas as pd

# Orijinal CSV dosyasını yükle
original_file_path = "fixed_shots/shots_fixed.csv"
df_original = pd.read_csv(original_file_path)

# 0021500491 numaralı maça ait verileri filtrele
df_filtered = df_original[df_original["GAME_ID"] == 21500491]

# Başarılı şutları filtrele (SHOT_ATTEMPTED_FLAG = 1 ve SHOT_MADE_FLAG = 1 olanlar)
df_made_shots = df_filtered[(df_filtered["SHOT_ATTEMPTED_FLAG"] == 1) & (df_filtered["SHOT_MADE_FLAG"] == 1)]

# Şutları zaman sırasına göre sırala
df_made_shots = df_made_shots.sort_values(by=["PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING"], ascending=[True, False, False])

# Ev sahibi ve rakip takımı belirle
home_team = df_filtered.iloc[0]["TEAM_NAME"]
away_team = df_filtered[df_filtered["TEAM_NAME"] != home_team].iloc[0]["TEAM_NAME"]

# Takımların başlangıç skorları
score_dict = {home_team: 0, away_team: 0}

# Skor değişimlerini takip eden yeni bir liste
score_progression = []

# Skor değişimlerini hesapla
for _, row in df_made_shots.iterrows():
    team = row["TEAM_NAME"]
    
    # Şutun 2'lik mi 3'lük mü olduğunu belirle
    points = 3 if "3PT" in row["SHOT_TYPE"] else 2
    
    # Skoru güncelle
    score_dict[team] += points

    # Kalan süreyi **doğru** şekilde saniye cinsine çevir
    remaining_time_seconds = row["MINUTES_REMAINING"] * 60 + row["SECONDS_REMAINING"]

    # O anki skoru kaydet (ondalık olmadan)
    score_progression.append(f"{row['PERIOD']},{int(remaining_time_seconds)},{score_dict[home_team]},{score_dict[away_team]}")

# Dosyayı kaydet (başlıksız format)
score_table_with_progress_path = "fixed_shots/score_result.csv"
with open(score_table_with_progress_path, "w") as file:
    file.write("\n".join(score_progression))
