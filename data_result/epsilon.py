import re
import matplotlib.pyplot as plt

# Read log data from log.txt
episodes = []
epsilon_home = []
epsilon_away = []

with open('log.txt', 'r') as file:
    for line in file:
        # Regex pattern to extract episode and epsilon values
        match = re.search(r'Episode\s+(\d+)/\d+.?Epsilon Home:\s([0-9.]+),\s*Epsilon Away:\s*([0-9.]+)', line)
        if match:
            episodes.append(int(match.group(1)))
            epsilon_home.append(float(match.group(2)))
            epsilon_away.append(float(match.group(3)))

# Plotting
plt.figure()
plt.plot(episodes, epsilon_home, label='Epsilon Home')
plt.plot(episodes, epsilon_away, label='Epsilon Away')
plt.xlabel('Episode')
plt.ylabel('Epsilon Value')
plt.title('Epsilon Decay from log.txt')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()