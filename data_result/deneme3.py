import numpy as np
import random
import math
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ 1. Basketbol Ortamı ------------------ #
class BasketballEnv:
    def __init__(self):
        self.state_dim = 29
        self.action_dim = 5
        self.reset()

    def reset(self):
        self.state = np.array([
            1.0, 720.0, 24.0,           # Periyot, süre, atak süresi
            47.0, 25.0, 10.0,           # Top konumu (x,y,z)
            *[np.random.uniform(0, 47) for _ in range(10)],  # Ev sahibi oyuncular
            *[np.random.uniform(47, 94) for _ in range(10)], # Rakip oyuncular
            1.0, 0.0, 0.0               # Top kontrolü, ev skor, rakip skor
        ], dtype=np.float32)
        return self.state.copy()

    def get_valid_actions(self, ball_control):
        if ball_control == 1:  # Ev sahibi
            return [0, 1, 3, 4]  # pas, şut, dribble, bekle
        return [2, 4]           # savunma, bekle

    def _update_ball_control(self, state):
        ball_pos = state[3:5]
        home_pos = np.array([[state[i], state[i+1]] for i in range(6, 16, 2)])
        away_pos = np.array([[state[i], state[i+1]] for i in range(16, 26, 2)])
        
        home_dists = np.linalg.norm(home_pos - ball_pos, axis=1)
        away_dists = np.linalg.norm(away_pos - ball_pos, axis=1)
        
        if min(home_dists) < 3 and min(home_dists) < min(away_dists):
            state[26] = 1  # Ev kontrolü
        elif min(away_dists) < 3:
            state[26] = 2  # Rakip kontrolü

    def step(self, action, home):
        new_state = self.state.copy()
        ball_control = new_state[26]
        reward = 0
        done = False

        # Süre güncelleme
        new_state[1] -= 0.2
        new_state[2] -= 0.2

        # Aksiyon işleme
        if ball_control == 1:  # Ev sahibi
            if action == 0:    # Pas
                if random.random() < 0.7:
                    new_state[3:5] += np.random.uniform(-5, 5, 2)
                    reward = 0.1
                else:
                    reward = -0.2
            elif action == 1:  # Şut
                distance = np.linalg.norm([new_state[3]-94, new_state[4]-25])
                p_success = 0.5 if distance < 20 else 0.35 if distance < 25 else 0.01
                if random.random() < p_success:
                    points = 2 if distance < 20 else 3
                    if home:
                        new_state[27] += points
                    else:
                        new_state[28] += points
                    reward = 1.0
        else:  # Rakip
            if action == 2:    # Savunma
                if random.random() < 0.4:
                    new_state[3:5] = np.random.uniform(30, 60, 2)
                    reward = -0.5  # Rakip başarılı savunma

        # Oyuncu hareketleri
        for i in range(6, 26):
            new_state[i] += np.random.uniform(-1.5, 1.5)
        
        # Sınır kontrolleri
        new_state[3] = np.clip(new_state[3], 0, 94)
        new_state[4] = np.clip(new_state[4], 0, 50)
        
        # Top kontrolü güncelleme
        self._update_ball_control(new_state)
        
        # Oyun sonu
        done = (new_state[1] <= 0) 
        self.state = new_state.copy()
        return new_state, reward, done, {"ball_control": ball_control}

# ------------------ 2. DQN Modeli ------------------ #
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ------------------ 3. Çift Epsilonlu Ajan ------------------ #
class DualEpsilonAgent:
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon_home = 0.1
        self.epsilon_away = 0.7
        self.is_pretraining = False
        self.action_dim = action_dim

    def load_offline_data(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            action_map = {"pass":0, "shot":1, "defend":2, "dribble":3, "":4}
            
            for state, action_str, reward in data:
                action = action_map.get(action_str, 4)
                state_tensor = torch.FloatTensor(state).to(device)
                self.memory.append((state_tensor, action, reward, state_tensor, True))
            
            print(f"Offline veri yüklendi: {len(data)} örnek")
            self.is_pretraining = True
        except Exception as e:
            print("Offline veri hatası:", e)

    def pretrain(self, batch_size=32, steps=1000):
        print("Offline pretraining başladı...")
        for step in range(steps):
            if len(self.memory) < batch_size:
                break
                
            batch = random.sample(self.memory, batch_size)
            states = torch.stack([x[0] for x in batch])
            actions = torch.LongTensor([x[1] for x in batch]).to(device)
            rewards = torch.FloatTensor([x[2] for x in batch]).to(device)
            
            pred_q = self.model(states).gather(1, actions.unsqueeze(1))
            loss = F.mse_loss(pred_q.squeeze(), rewards)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (step+1) % 200 == 0:
                print(f"Pretrain Step {step+1}/{steps} | Loss: {loss.item():.4f}")
        
        self.is_pretraining = False
        print("Offline pretraining tamamlandı!")

    def get_action(self, state, is_home_team):
        if self.is_pretraining:
            return random.choice(range(self.action_dim))
            
        epsilon = self.epsilon_home if is_home_team else self.epsilon_away
        valid_actions = env.get_valid_actions(state[26])
        
        if random.random() < epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.model(state_tensor)[0].cpu().numpy()
            q_values[[a for a in range(self.action_dim) if a not in valid_actions]] = -np.inf
            return np.argmax(q_values)

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        
        # Tensor işlemleri düzeltildi
        states = torch.stack([x[0].cpu() if x[0].is_cuda else x[0] for x in batch]).to(device)
        actions = torch.LongTensor([x[1] for x in batch]).to(device)
        rewards = torch.FloatTensor([x[2] for x in batch]).to(device)
        next_states = torch.stack([x[3].cpu() if x[3].is_cuda else x[3] for x in batch]).to(device)
        dones = torch.BoolTensor([x[4] for x in batch]).to(device)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.model(next_states).max(1)[0].detach()
        targets = rewards + (self.gamma * next_q * ~dones)
        
        loss = F.mse_loss(current_q.squeeze(), targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ------------------ 4. Eğitim ve Test ------------------ #
if __name__ == "__main__":
    # Ortam ve ajan
    env = BasketballEnv()
    agent = DualEpsilonAgent(env.state_dim, env.action_dim)
    
    # 1. Offline pretraining
    agent.load_offline_data("Last_result_data/21500001_last_result.json")
    agent.pretrain(steps=1000)
    
    # 2. Online eğitim
    print("\nOnline eğitim başlıyor...")
    for episode in range(10):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Çift aksiyon seçimi
            home_action = agent.get_action(state, is_home_team=True)
            away_action = agent.get_action(state, is_home_team=False)
            ball_control = state[26]

            if ball_control == 1:
                next_state, reward, done, info = env.step(home_action ,home=True)
                total_reward += reward
            else:
                for i in range(6, 26, 2):
                    env.state[i] = 47 * 2 - env.state[i]  # x pozisyonunu simetrik yap
                next_state, reward, done, info = env.step(away_action ,home=False)
                for i in range(6, 26, 2):
                    next_state[i] = 47 * 2 - next_state[i]  # x pozisyonunu simetrik yap
                total_reward += reward
            # Ortam güncelleme

            
            # Sadece ev sahibi deneyimlerini kaydet
            agent.memory.append((
                torch.FloatTensor(state).to(device),
                home_action,
                reward,
                torch.FloatTensor(next_state).to(device),
                done
            ))
            
            # Online eğitim
            agent.train(batch_size=32)
            state = next_state
        
        # Epsilon decay
        agent.epsilon_home = max(0.01, agent.epsilon_home * 0.995)
        
        if (episode+1) % 1 == 0:
            print(f"Episode {episode+1} | Reward: {total_reward:.1f} | "
                  f"Eps: Home={agent.epsilon_home:.3f}, Away={agent.epsilon_away:.1f} | "
                  f"Score: {env.state[27]:.0f}-{env.state[28]:.0f}")
    
    # 3. Test
    print("\nTest modu:")
    state = env.reset()
    done = False
    while not done:
        home_action = agent.get_action(state, is_home_team=True)
        state, _, done, _ = env.step(home_action)
        print(f"Top: [{state[3]:.1f}, {state[4]:.1f}] | Skor: {state[27]}-{state[28]} | Kontrol: {'Ev' if state[26] == 1 else 'Rakip'}")
    print(f"Final Skor: {state[27]}-{state[28]}")