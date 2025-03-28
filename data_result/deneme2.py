import json
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# ------------------ 1. Basketbol Ortamı ------------------ #
class BasketballEnv:
    def __init__(self):
        self.state_dim = 29  # 29 elemanlı durum uzayı
        self.action_dim = 5  
        self.reset()

    def reset(self):
        """Oyunu sıfırlar ve başlangıç durumunu döndürür."""
        self.state = [
            1.0,  # Periyot
            720.0,  # Kalan periyot süresi
            24.0,  # Kalan atak süresi
            47.0,  # Top X konumu (Sahanın ortası)
            25.0,  # Top Y konumu
            10.0   # Top Z konumu
        ]

        # Ev sahibi oyuncular (5 oyuncu, x: 0-47, y: 0-50)
        for _ in range(5):
            x = np.random.randint(0, 47)
            y = np.random.randint(0, 50)
            self.state.append(x)
            self.state.append(y)

        # Rakip oyuncular (5 oyuncu, x: 47-94, y: 0-50)
        for _ in range(5):
            x = np.random.randint(47, 94)
            y = np.random.randint(0, 50)
            self.state.append(x)
            self.state.append(y)

        # **Skor Verileri**
        self.state.append(0.0)  # Topa sahip takımın skoru
        self.state.append(0)    # Ev sahibi skor
        self.state.append(0)    # Deplasman skor

        return self.state

    def step(self, action):
        """
        Ajanın seçtiği aksiyona göre yeni durumu ve ödülü döndürür.
        """
        new_state = self.state.clone()

        # **Süre azalımı**
        new_state[1] -= 0.2  # Periyot süresi azalır
        new_state[2] -= 0.2  # Atak süresi azalır

        if action == 0:
            success = random.random() < 0.7  # %70 başarılı pas ihtimali

            if success:
                # **Başarılı Pas:**
                new_state[3] -= 4.0  # X ekseni değişimi
                new_state[4] += 8.0  # Y ekseni değişimi
                new_state[5] += 1.0  # Z ekseni değişimi

                # Oyuncu konumlarını değiştir
                for i in range(6, 26, 2):  
                    new_state[i] += np.random.uniform(-2, 2)
                    new_state[i+1] += np.random.uniform(-2, 2)

                reward = 0.1

            else:
                # **Başarısız Pas:**
                new_state[3] += 2.0
                new_state[4] += 2.0
                new_state[5] += 1.0

                for i in range(6, 26, 2):  
                    new_state[i] += np.random.uniform(-2, 2)
                    new_state[i+1] += np.random.uniform(-2, 2)

                reward = -0.2

        elif action == 1:
            distance = math.sqrt((new_state[3] - 88)**2 + (new_state[4] - 25)**2)
            if distance < 20:
                success = random.random() < 0.50  # **%47 başarılı şut ihtimali**
                if success:
                    # **Başarılı Şut:**
                    new_state[3] = 0.0  
                    new_state[4] = random.uniform(10, 20)
                    new_state[5] = 5.0  

                    # **Ev sahibi skor 2 veya 3 artırılır**
                    points = 2
                    new_state[27] += points  
                    reward = 1  

                else:
                    # **Başarısız Şut:**
                    new_state[3] += 5.0  
                    new_state[4] += 5.0  
                    new_state[5] = 5.0  

                    reward = 0 
            elif distance < 25:
                success = random.random() < 0.35
                if success:
                    # **Başarılı Şut:**
                    new_state[3] = 0.0  
                    new_state[4] = random.uniform(10, 20)
                    new_state[5] = 5.0  

                    # **Ev sahibi skor 2 veya 3 artırılır**
                    points = 3
                    new_state[27] += points  
                    reward = 1  

                else:
                    # **Başarısız Şut:**
                    new_state[3] += 5.0  
                    new_state[4] += 5.0  
                    new_state[5] = 5.0  

                    reward = 0 
            else:
                success = random.random() < 0.01
                if success:
                    # **Başarılı Şut:**
                    new_state[3] = 0.0  
                    new_state[4] = random.uniform(10, 20)
                    new_state[5] = 5.0  

                    # **Ev sahibi skor 2 veya 3 artırılır**
                    points = 2
                    new_state[27] += points  
                    reward = 1  

                else:
                    # **Başarısız Şut:**
                    new_state[3] += 5.0  
                    new_state[4] += 5.0  
                    new_state[5] = 5.0  

                    reward = 0  

            # **Oyuncuların konumu şuta göre değişiyor**
            for i in range(6, 26, 2):  
                new_state[i] += np.random.uniform(-2, 2)
                new_state[i+1] += np.random.uniform(-2, 2)

        elif action == 2:
            success = random.random() < 0.4  # **%40 başarılı savunma ihtimali**

            if success:
                # **Başarılı Savunma:**
                new_state[3] = random.uniform(10, 30)  # Top X konumu değişir
                new_state[4] = random.uniform(15, 35)  # Top Y konumu değişir
                new_state[5] = random.uniform(2, 4)  # Hafif Z değişimi

                # **Savunan oyuncular pozisyon alıyor**
                for i in range(6, 16, 2):  # Ev sahibi oyuncular (5 oyuncu)
                    new_state[i] += np.random.uniform(-3, 3)
                    new_state[i+1] += np.random.uniform(-3, 3)

                reward = 0.2  

            else:
                # **Başarısız Savunma:**
                new_state[3] += np.random.uniform(1, 3)  
                new_state[4] += np.random.uniform(1, 3)  
                new_state[5] = 3.0  

                for i in range(6, 26, 2):  
                    new_state[i] += np.random.uniform(-2, 2)
                    new_state[i+1] += np.random.uniform(-2, 2)

                reward = 0  

        elif action == 3:
            # **Tek İhtimalle Dribble**
            # Dribbling aksiyonu ile topun pozisyonu değişiyor
            new_state[3] += np.random.uniform(1, 2)  # Top X konumu değişimi
            new_state[4] += np.random.uniform(1, 2)  # Top Y konumu değişimi
            new_state[5] += np.random.uniform(0, 1)  # Top Z konumu değişimi

            # **Oyuncu pozisyonları biraz kayar**
            for i in range(6, 26, 2):  
                new_state[i] += np.random.uniform(-2, 2)
                new_state[i+1] += np.random.uniform(-2, 2)

            reward = 0  # Dribbling aksiyonunda ödül yok

        elif action == 4:
            # **Boş Aksiyon ("")**
            # Topun ve oyuncuların konumu yalnızca çok küçük bir değişim gösterir
            new_state[3] += np.random.uniform(0, 1)  # Top X konumu
            new_state[4] += np.random.uniform(0, 1)  # Top Y konumu
            new_state[5] += np.random.uniform(0, 0.5)  # Top Z konumu

            # **Oyuncu pozisyonları biraz kayar**
            for i in range(6, 26, 2):  
                new_state[i] += np.random.uniform(-1, 1)
                new_state[i+1] += np.random.uniform(-1, 1)

            reward = 0  # Boş aksiyonla ödül verilmez
        else:
            print("hatalı action")
            print (action)
  
        # Oyun bitti mi?
        #print(new_state[1])
        #print(reward)
        done = new_state[1] <= 0 

        return new_state, reward, done

# ------------------ 2. DQN Ajan Modeli ------------------ #
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ------------------ 3. Offline Verinin Yüklenmesi ------------------ #
def load_offline_data(file_path):
    """
    Offline verinizin her örneği [state, action, reward] formatında olmalıdır.
    Örneğin:
    [
      [[...29 elemanlı state...], "shot", 2],
      [[...29 elemanlı state...], "pass", 10],
      ...
    ]
    """
    action_map = {
        "pass": 0,
        "shot": 1,
        "defend": 2,
        "dribble": 3,
        "": 4  # boş aksiyon
    }

    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Eylemleri sayısal değerlere dönüştür
    for sample in data:
        state, action, reward = sample
        action = action_map.get(action, 4)  # Eğer tanımlanmış değilse, varsayılan olarak 4 (boş aksiyon) kabul edilir
        sample[1] = action  # Aksiyonu sayısal değere dönüştürüp güncelle
    
    return data

# ------------------ 4. Eğitim Parametreleri ------------------ #
env = BasketballEnv()
state_dim = env.state_dim
action_dim = env.action_dim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn = DQN(state_dim, action_dim).to(device)
optimizer = optim.Adam(dqn.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
memory = deque(maxlen=10000)

gamma = 0.95  
epsilon = 1.0  
epsilon_min = 0.01  
epsilon_decay = 0.995  
batch_size = 32  

# ------------------ 5. Offline Pretraining ------------------ #
# Offline veriyi yükle ve replay buffer'a ekle
offline_data_path = "with_passes491.json"  # Offline verinizi içeren dosya
try:
    offline_data = load_offline_data(offline_data_path)
    for sample in offline_data:
        # Her örnek: [state, action, reward]
        state, action, reward = sample
        state_tensor = torch.FloatTensor(state).to(device)
        # Offline veride next_state bilgisi olmadığı için mevcut state kullanılıyor
        next_state_tensor = torch.FloatTensor(state).to(device)
        # Tek adımlık deneyim olarak kabul edip terminal durum (done=True) atıyoruz
        done = True
        memory.append((state_tensor, action, reward, next_state_tensor, done))
    print(f"Offline veriden {len(offline_data)} deney replay buffer'a eklendi.")
except Exception as e:
    print("Offline veri yüklenemedi:", e)

# Offline pretraining için belirli adım sayısı (örneğin 1000 iterasyon)
pretrain_steps = 1
for step in range(pretrain_steps):
    if len(memory) < batch_size:
        break  # Yeterli veri yoksa pretrain yapılamaz
    
    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states = torch.stack(states)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.stack(next_states)
    dones = torch.BoolTensor(dones).to(device)

    q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    # Her örnek terminal kabul edildiği için target Q değeri sadece ödül
    target_q_values = rewards

    loss = loss_fn(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (step + 1) % 200 == 0:
        print(f"Offline pretraining step: {step + 1}/{pretrain_steps}")


# ------------------ 6. Online Eğitim ------------------ #
episodes = 25  # Online eğitim için epizod sayısı
for episode in range(episodes):
    state = env.reset()
    env.state = torch.FloatTensor(state).to(device)  # state'i doğru cihaza taşı
    total_reward = 0
    done = False
    while not done:
        # ε-greedy stratejisi ile aksiyon seçimi
        if random.random() < epsilon:
            action = random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                q_values = dqn(env.state)
                action = torch.argmax(q_values).item()

        next_state, reward, done = env.step(action)  # Bu kısımda yeni state alınır
        next_state = next_state.to(device)  # Cihaza taşıma işlemi (önce CPU'ya gerek yok)

        memory.append((env.state, action, reward, next_state, done))  # Memory'e ekle
        env.state = next_state  # State güncellenir (oyun bir sonraki adıma geçer)
        total_reward += reward

        # Online öğrenme adımı
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)

            states = torch.stack(states).to(device)  # Cihaza taşımayı unutma
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.stack(next_states).to(device)  # Cihaza taşımayı unutma
            dones = torch.BoolTensor(dones).to(device)

            q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = dqn(next_states).max(1)[0].detach()
            target_q_values = rewards + (gamma * next_q_values * ~dones)

            loss = loss_fn(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    

    # Epsilon değerini azalt
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}")
    print(env.state)


# ------------------ 7. Eğitim Sonrası Test ------------------ #
print("\nEğitim Tamamlandı! Model Test Ediliyor...\n")
for test in range(5):
    state = env.reset()
    env.state = torch.FloatTensor(state).to(device)  # İlk state'i cihaza taşı
    done = False
    total_reward = 0
    while not done:
        with torch.no_grad():
            q_values = dqn(env.state)
            action = torch.argmax(q_values).item()  # Maksimum Q değerine sahip aksiyonu seç

        next_state, reward, done = env.step(action)  # Aksiyon sonrası yeni state ve ödül alınır
        next_state = next_state.to(device)  # Yeni state'i cihaza taşı

        env.state = next_state  # State güncellenir
        total_reward += reward  # Toplam ödül biriktirilir

    print(f"Test {test + 1}: Toplam Ödül = {total_reward}")

print(q_values)

