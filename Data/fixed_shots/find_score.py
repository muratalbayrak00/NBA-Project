import pandas as pd

# Dosya yolu
file_path = "Data/fixed_shots/shots_fixed.csv"

# CSV dosyasını yükle
df_original = pd.read_csv(file_path)

# İlgili maç ID'sini filtrele
game_id = 21500491
df_filtered = df_original[df_original["GAME_ID"] == game_id]

# Başarılı şutları filtrele (SHOT_ATTEMPTED_FLAG = 1 ve SHOT_MADE_FLAG = 1 olanlar)
df_made_shots = df_filtered[(df_filtered["SHOT_ATTEMPTED_FLAG"] == 1) & (df_filtered["SHOT_MADE_FLAG"] == 1)]

# Şutları zaman sırasına göre sırala
df_made_shots = df_made_shots.sort_values(by=["PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING"], ascending=[True, False, False])

# Ev sahibi ve rakip takımı belirle
home_team = df_filtered.iloc[0]["TEAM_NAME"]
away_team = df_filtered[df_filtered["TEAM_NAME"] != home_team].iloc[0]["TEAM_NAME"]

# Skor değişimlerini hesapla
score_progression = []

for _, row in df_made_shots.iterrows():
    team = row["TEAM_NAME"]
    points = 3 if "3PT" in row["SHOT_TYPE"] else 2  # 3'lük veya 2'lik şut kontrolü
    game_clock = row["MINUTES_REMAINING"] * 60 + row["SECONDS_REMAINING"]  # Oyun saati (saniye cinsinden)

    # Basketi atan takımın ev sahibi mi rakip mi olduğunu belirle
    is_home_team = 1 if team == home_team else 0

    # Yeni formatta listeye ekle
    score_progression.append(f"{row['PERIOD']},{int(game_clock)},{is_home_team},{points}")

# Güncellenmiş veriyi dosyaya kaydet
output_file_path = "Data/fixed_shots/score_result.csv"

with open(output_file_path, "w") as file:
    file.write("\n".join(score_progression))

# Dosya yolu döndürme
output_file_path

