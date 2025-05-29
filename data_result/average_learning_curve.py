import re
import matplotlib.pyplot as plt

log_path = "log.txt"  # Log dosyasının yolu

# Episode ve Reward verilerini oku
episodes = []
rewards = []

with open(log_path, 'r') as file:
    for line in file:
        match = re.search(r"Episode (\d+)/\d+, Reward: ([\-\d.]+)", line)
        #match = re.search(r"Episode (\d+)+, Reward: ([\-\d.]+)", line)
        if match:
            ep = int(match.group(1))
            rw = float(match.group(2))
            episodes.append(ep)
            rewards.append(rw)

# Kademeli + kayan ortalama hesapla (window size: 10)
window_size = 50
avg_rewards = []

for i in range(len(rewards)):
    start_idx = max(0, i - window_size + 1)
    window = rewards[start_idx:i + 1]
    avg_rewards.append(sum(window) / len(window))

# Grafiği çiz
plt.figure(figsize=(10, 5))
plt.plot(episodes, avg_rewards, color='darkorange', label='Moving average reward')
plt.xlabel("Episode Number")
plt.ylabel("Average Reward")
plt.title(f"Learning curve with a sliding window of {window_size} episodes")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
