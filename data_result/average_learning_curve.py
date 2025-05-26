import re
import matplotlib.pyplot as plt
import numpy as np

log_path = "log.txt"  # Log dosyasının adı

# Episode ve Reward verilerini toplayalım
episodes = []
rewards = []

with open(log_path, 'r') as f:
    for line in f:
        match = re.search(r"Episode (\d+)/\d+, Reward: ([\-\d.]+)", line)
        if match:
            ep = int(match.group(1))
            rw = float(match.group(2))
            episodes.append(ep)
            rewards.append(rw)

# Ortalama rewardları hesapla (örnek: her 10 bölümde bir ortalama)
window_size = 30
avg_episodes = []
avg_rewards = []

for i in range(0, len(rewards), window_size):
    window = rewards[i:i+window_size]
    if len(window) == window_size:
        avg_episodes.append(episodes[i + window_size // 2])  # orta noktayı kullan
        avg_rewards.append(np.mean(window))

# Öğrenme eğrisini çiz
plt.figure(figsize=(10, 5))
plt.plot(avg_episodes, avg_rewards, marker='o', linestyle='-', color='blue')
plt.xlabel("Episode")
plt.ylabel("Average Reward (per 10 episodes)")
plt.title("Learning Curve from Log File")
plt.grid(True)
plt.tight_layout()
plt.show()
