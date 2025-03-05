import json
import pandas as pd
import matplotlib.pyplot as plt

# JSON dosyasını okuma
with open("21500491_result.json", "r") as file:
    data = json.load(file)

# Tüm maç için veriyi işleme
home_scores_per_quarter = [0, 0, 0, 0]  # Ev sahibi takımın periyot bazında skorları
away_scores_per_quarter = [0, 0, 0, 0]  # Deplasman takımının periyot bazında skorları
home_passes_per_quarter = [0, 0, 0, 0]  # Ev sahibi takımın pas sayıları periyot bazında
away_passes_per_quarter = [0, 0, 0, 0]  # Deplasman takımının pas sayıları periyot bazında

quarters = [720, 540, 360, 180]  # Periyot başlama zamanları (12 dakika = 720 saniye)

current_home_score = 0
current_away_score = 0
current_home_passes = 0
current_away_passes = 0
current_quarter = 0  # Başlangıçta 1. periyot

for game in data:
    moments = game["moments"]
    for moment in moments:
        game_clock = moment["game_clock"]
        
        # Her periyotta takımların sayısını güncelle
        if moment.get("action") == "shot":
            if moment["ball_owner"]["team_id"] == 1610612748:  # Ev sahibi takım
                current_home_score += 2
            elif moment["ball_owner"]["team_id"] == 1610612742:  # Deplasman takımı
                current_away_score += 2
        
        # Pas eylemi
        if moment.get("action") == "pass":
            if moment["ball_owner"]["team_id"] == 1610612748:  # Ev sahibi takım
                current_home_passes += 1
            elif moment["ball_owner"]["team_id"] == 1610612742:  # Deplasman takımı
                current_away_passes += 1
        
        # Periyot değişimini kontrol et
        if game_clock <= quarters[current_quarter] and game_clock > quarters[current_quarter + 1] if current_quarter < 3 else game_clock > 0:
            home_scores_per_quarter[current_quarter] = current_home_score
            away_scores_per_quarter[current_quarter] = current_away_score
            home_passes_per_quarter[current_quarter] = current_home_passes
            away_passes_per_quarter[current_quarter] = current_away_passes
        else:
            current_quarter += 1

# Periyotları etiketleme
period_labels = [f"Q{i+1}" for i in range(4)]

# Skor Grafiği
fig1, ax1 = plt.subplots(figsize=(12, 8))
width = 0.35  # Sütunların genişliği
x = range(4)

# Ev sahibi ve deplasman takımları için skor sütunları
ax1.bar(x, home_scores_per_quarter, width, label="Home Team", color="lightblue")
ax1.bar([i + width for i in x], away_scores_per_quarter, width, label="Away Team", color="lightcoral")

# Grafik başlığı ve etiketleri
ax1.set_title("Team Scores Per Quarter", fontsize=18)
ax1.set_xlabel("Quarter", fontsize=14)
ax1.set_ylabel("Score", fontsize=14)
ax1.set_xticks([i + width / 2 for i in x])
ax1.set_xticklabels(period_labels, fontsize=12)
ax1.set_yticks(range(0, max(home_scores_per_quarter + away_scores_per_quarter) + 5, 5))
ax1.tick_params(axis='y', labelsize=12)
ax1.legend(fontsize=12)

# Pas Grafiği
fig2, ax2 = plt.subplots(figsize=(12, 8))
# Pas sayıları için sütunlar
ax2.bar(x, home_passes_per_quarter, width, label="Home Passes", color="blue", alpha=0.5)
ax2.bar([i + width for i in x], away_passes_per_quarter, width, label="Away Passes", color="red", alpha=0.5)

# Grafik başlığı ve etiketleri
ax2.set_title("Team Passes Per Quarter", fontsize=18)
ax2.set_xlabel("Quarter", fontsize=14)
ax2.set_ylabel("Passes", fontsize=14)
ax2.set_xticks([i + width / 2 for i in x])
ax2.set_xticklabels(period_labels, fontsize=12)
ax2.set_yticks(range(0, max(home_passes_per_quarter + away_passes_per_quarter) + 5, 5))
ax2.tick_params(axis='y', labelsize=12)
ax2.legend(fontsize=12)

# Grafik gösterimi
plt.tight_layout()
plt.show()
