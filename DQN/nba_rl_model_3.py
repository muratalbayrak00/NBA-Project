import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.model_selection import train_test_split  # Veriyi bölmek için

# DQN Modeli
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# JSON dosyasını yükleme
def load_match_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Veriyi işleme ve replay buffer'ı oluşturma
def process_data(data):
    states = []
    actions_idx = []
    rewards = []

    # Eylemleri indekslere dönüştürmek için sözlük
    actions = ["defend_fail", "succesfull_defend", "idle_defend", "dribble", "succesfull_shot","missed_shot","succesfull_pass","missed_pass","idle_pass","idle"]
    action_to_idx = {action: idx for idx, action in enumerate(actions)}

    for entry in data:
        state = entry[0]  # Durum vektörü
        action = entry[1][0]
        reward = entry[2]  # Ödül
        

        # Eylemi indekse dönüştür
        action_idx = action_to_idx.get(action, -1)  # Bilinmeyen eylemler için -1 döner
        if action_idx == -1:
            print(f"Bilinmeyen eylem: {action}")
            continue  # Bilinmeyen eylemleri atla

        states.append(state)
        actions_idx.append(action_idx)
        rewards.append(reward)

    # NumPy array'lerine dönüştürme
    states = np.array(states, dtype=np.float32)
    actions_idx = np.array(actions_idx, dtype=np.int64)
    for i, reward in enumerate(rewards):
        if isinstance(reward, (list, tuple)):
            print(f"Index {i}: {reward} is a sequence!")
    rewards = np.array(rewards, dtype=np.float32)

    return states, actions_idx, rewards

# Replay buffer'ı oluşturma
def create_replay_buffer(states, actions_idx, rewards):
    replay_buffer = []

    for i in range(len(states) - 1):
        state = states[i]
        action_idx = actions_idx[i]
        reward = rewards[i]
        next_state = states[i + 1]
        done = 0  # Oyunun bitip bitmediğini belirten bir flag (örneğin, son momentte 1 olabilir)

        replay_buffer.append((state, action_idx, reward, next_state, done))

    return replay_buffer

# DQN Eğitimi
def train_dqn(model, target_model, optimizer, loss_fn, replay_buffer, batch_size=32, gamma=0.99):
    if len(replay_buffer) < batch_size:
        return

    batch = random.sample(replay_buffer, batch_size)
    states, actions_idx, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions_idx = torch.tensor(actions_idx, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = model(states).gather(1, actions_idx)
    next_q_values = target_model(next_states).max(1, keepdim=True)[0]
    target_q_values = rewards + (1 - dones) * gamma * next_q_values

    loss = loss_fn(q_values, target_q_values.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Modeli test verisi üzerinde değerlendirme
def evaluate_model(model, test_states, test_actions_idx):
    correct_predictions = 0
    total_predictions = len(test_states)

    for i in range(total_predictions):
        state = test_states[i]
        true_action_idx = test_actions_idx[i]

        # Modelin tahminini al
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = model(state_tensor)
            predicted_action_idx = torch.argmax(q_values).item()

        # Tahminin doğru olup olmadığını kontrol et
        if predicted_action_idx == true_action_idx:
            correct_predictions += 1

    # Doğruluk hesapla
    accuracy = correct_predictions / total_predictions
    return accuracy

# Test verisi üzerinde toplam ödül hesaplama
def calculate_test_total_reward(model, test_states, test_actions_idx, test_rewards):
    total_reward = 0

    for i in range(len(test_states)):
        state = test_states[i]
        true_action_idx = test_actions_idx[i]
        reward = test_rewards[i]

        # Modelin tahminini al
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = model(state_tensor)
            predicted_action_idx = torch.argmax(q_values).item()

        # Tahmin edilen eylemin ödülünü topla
        if predicted_action_idx == true_action_idx:
            total_reward += reward

    return total_reward

# Ana program
if __name__ == "__main__":
    # Dosya yolunu platform bağımsız olarak belirle
    file_path = os.path.join("../Data/find_action", "match_data_493.json")
    print("Dosya Yolu:", file_path)

    # JSON dosyasını yükle
    data = load_match_data(file_path)

    # Veriyi train ve test olarak ayırma
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Train ve test verisini işle
    train_states, train_actions_idx, train_rewards = process_data(train_data)
    test_states, test_actions_idx, test_rewards = process_data(test_data)

    # Replay buffer'ı oluştur
    replay_buffer = create_replay_buffer(train_states, train_actions_idx, train_rewards)

    # Model ve optimizasyon parametreleri
    state_dim = len(train_states[0])  # Durum vektörünün boyutu
    action_dim = 10  # Eylem sayısı (örneğin, 10 farklı eylem)
    model = DQN(state_dim, action_dim)
    target_model = DQN(state_dim, action_dim)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    gamma = 0.99

    # Eğitim döngüsü
    num_episodes = 1  # Eğitim episode sayısı
    for episode in range(num_episodes):
        total_reward = 0
        for i in range(len(replay_buffer)):
            train_dqn(model, target_model, optimizer, loss_fn, replay_buffer, gamma=gamma)
            state, action_idx, reward, next_state, done = replay_buffer[i]
            total_reward += reward


        print(f"Episode {episode + 1} completed. Total Reward: {total_reward:.2f}")
        

    # Modeli test verisi üzerinde değerlendir
    accuracy = evaluate_model(model, test_states, test_actions_idx)
    print(f"Test Verisi Üzerinde Doğruluk: {accuracy * 100:.2f}%")

    # Modeli kaydet (isteğe bağlı)
    torch.save(model.state_dict(), "dqn_model.pth")  