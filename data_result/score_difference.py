import re
import matplotlib.pyplot as plt

# Log dosyasını oku
with open('log3.txt', 'r') as file:
    log_lines = file.readlines()

# Verileri ayrıştır
episodes = []
score_differences = []  # Home - Away

for line in log_lines:
    match = re.search(r"Episode (\d+),.*Score: (\d+)-(\d+)", line)
    if match:
        episode = int(match.group(1))
        home = int(match.group(2))
        away = int(match.group(3))
        episodes.append(episode)
        score_differences.append(home - away)

# Grafik çizimi (sadece scatter/noktalar)
plt.figure(figsize=(10, 6))
plt.scatter(episodes, score_differences, color='purple', label='Home - Away Score')
plt.axhline(0, color='gray', linestyle='--')  # Sıfır çizgisi
plt.xlabel('Episode')
plt.ylabel('Score Difference (Home - Away)')
plt.title('Score Difference per Episode (Only Dots)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
