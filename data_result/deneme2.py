import json
import os
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
    # BasketballEnv sınıfına bu sabitleri ekleyin
    COURT_MIN_X = 0
    COURT_MAX_X = 94
    COURT_MIN_Y = 0
    COURT_MAX_Y = 50
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
        self.state.append(1)  # Topa sahip takımın skoru
        self.state.append(0)    # Ev sahibi skor
        self.state.append(0)    # Deplasman skor

        return self.state
    
    def get_valid_actions(self, is_home_team):
        """Top kontrolüne göre geçerli aksiyonları döndürür"""
        """ball_control = self.state[26]
        
        if is_home_team:
            if ball_control == 1:  # Ev sahibi topa sahip
                return [0, 1, 3, 4]  # pas, şut, dribble, ""
            else:
                return [2, 4]  # savunma, ""
        else:  # Rakip takım
            if ball_control == 2:  # Rakip topa sahip
                return [0, 1, 3, 4]  # pas, şut, dribble
            else:
                return [2, 4]  # savunma, bekle"""
        return [0, 1, 3, 4]
    
    def select_pass_target(self, state, ball_x, ball_y):
        ball_owner = int(state[26])  # Topa sahip takım (0: ev sahibi, 1: deplasman)

        # İlgili takımın oyuncularının index aralığını belirle
        if ball_owner == 1:
            start_idx = 6
            end_idx = 16
        else:
            start_idx = 6
            end_idx = 16

        candidates = []

        # Adayları belirle: topun ilerisinde olanlar
        for i in range(start_idx, end_idx, 2):
            x = state[i]
            y = state[i+1]
            distance = np.linalg.norm([(x - ball_x).cpu().numpy(), (y - ball_y).cpu().numpy()])
            if y > ball_y:  # sadece topun ilerisindekileri al
                candidates.append((distance, i))

        # Eğer topun önünde kimse yoksa, en yakın oyuncuyu seç
        if not candidates:
            closest = None
            min_distance = float('inf')
            for i in range(start_idx, end_idx, 2):
                x = state[i]
                y = state[i+1]
                distance = np.linalg.norm([(x - ball_x).cpu().numpy(), (y - ball_y).cpu().numpy()])
                if distance < min_distance:
                    min_distance = distance
                    closest = i
            return closest

        # Eğer birden fazla aday varsa, olasılıklı seçim yap
        candidates.sort()  # mesafeye göre sırala (yakından uzağa)

        if len(candidates) == 1:
            return candidates[0][1]  # sadece bir aday varsa onu döndür

        # %70 ihtimalle en yakın olanı, %30 ihtimalle ikinci en yakın olanı döndür
        rand = random.random()
        if rand < 0.7:
            return candidates[0][1]
        else:
            return candidates[1][1]
        
    def clamp_position(self, x, y, z=None):
        """Pozisyonu saha sınırları içinde tutar"""
        x = max(self.COURT_MIN_X, min(self.COURT_MAX_X, x))
        y = max(self.COURT_MIN_Y, min(self.COURT_MAX_Y, y))
        if z is not None:
            z = max(0, z)  # Top zeminden aşağıda olamaz
            return x, y, z
        return x, y
    
    def update_player_positions(self, state, start_index, end_index, move_range):
        for i in range(start_index, end_index, 2):
            new_x = state[i] + np.random.uniform(-move_range, move_range)
            new_y = state[i+1] + np.random.uniform(-move_range, move_range)
            state[i], state[i+1] = self.clamp_position(new_x, new_y)

    def step(self, action):
        
        new_state = self.state.clone()

        # **Süre azalımı**
        new_state[1] -= 0.2  # Periyot süresi azalır
        new_state[2] -= 0.2  # Atak süresi azalır

        ball_x = new_state[3]
        ball_y = new_state[4]

        if action == 0:  # PASS
            target_index = self.select_pass_target(new_state, ball_x, ball_y)
            if target_index is not None: # TODO: none donmuyor zaten hic o yuzden else kismi olmasada olur gibi
                pass_success = random.random() < 0.83
                if pass_success:
                    target_x = new_state[target_index]
                    target_y = new_state[target_index + 1]
                    # Topu hedef oyuncuya 3 step'te yaklaştırıyoruz
                    # TODO: burasi su an istedigimiz gibi calismiyor cunku next stati alamsdigim icin next stati guncelleyemiyoruz sadece o anki statede degisiklik yapiyoruz. ileride duzenlenmeli 
                    for _ in range(3):  # Her 0.2s için #TODO: burasi degistirileblir topun alitacagi oyuncunun uzakligina gore yapilir uzaksa 3 4 adimda yakinsa 1 2 adimda bas atilabilir. 
                        new_state[3] += (target_x - new_state[3]) * 0.33
                        new_state[4] += (target_y - new_state[4]) * 0.33
                        self.update_player_positions(new_state, 6, 16, 2.0)
                        self.update_player_positions(new_state, 16, 26, 2.0)
                    reward = 0.1
                else:
                    # Başarısız pas: top rastgele yere gider
                    new_state[3] += np.random.uniform(-5, 5)
                    new_state[4] += np.random.uniform(-5, 5)
                    new_state[26] = 3-new_state[26]
                    reward = -0.2
            else:
                new_state[3] += np.random.uniform(-5, 5)
                new_state[4] += np.random.uniform(-5, 5)
                new_state[26] = 3-new_state[26]
                # Pas atılacak kimse yok
                reward = -0.5 # TODO: degistir

        elif action == 1:  # SHOT
            # Potanın konumunu belirle (topa sahip olan takımın hücum yaptığı pota)
            ball_owner = int(new_state[26])
            #hoop_x = 88 if ball_owner == 1 else 5
            #hoop_y = 25
            hoop_x = 88
            hoop_y = 25

            # Potaya uzaklık
            distance = math.sqrt((new_state[3] - hoop_x)**2 + (new_state[4] - hoop_y)**2)

            # Rakip oyunculara (16–26 arası veya 6–16 arası) olan minimum mesafe
            defender_start = 16 
            defender_end = 26 
            min_defender_distance = float('inf')

            for i in range(defender_start, defender_end, 2):
                dx = new_state[3] - new_state[i]
                dy = new_state[4] - new_state[i+1]
                d = math.sqrt(dx**2 + dy**2)
                if d < min_defender_distance:
                    min_defender_distance = d

            # Başarı olasılığı hesapla
            if distance < 20:
                base_prob = 0.5
                points = 2
            elif distance < 25:
                base_prob = 0.35
                points = 3
            else:
                base_prob = 0.01
                points = 3 if distance >= 25 else 2

            # Savunma etkisi: Yakın savunma varsa başarı oranı düşer
            if min_defender_distance < 2:
                base_prob -= 0.15
            elif min_defender_distance < 5:
                base_prob -= 0.05

            success = random.random() < base_prob

            if success:
                #  Skor güncellenir
                if ball_owner == 1:
                    new_state[27] += points  # Ev sahibi
                elif ball_owner == 2:
                    new_state[28] += points  # Deplasman

                reward = 1

                # Top rakibe geçer
                new_state[26] = 3-new_state[26] # ball_owner değişir

                #  Top yeni ball_owner takımının PG oyuncusuna (örnek: ilk oyuncu) geçer
                new_ball_owner_team_start = 16 
                new_state[3] = new_state[new_ball_owner_team_start]      # Top x
                new_state[4] = new_state[new_ball_owner_team_start + 1]  # Top y
                new_state[5] = 5.0  # Top z

                #  Atak süresi sıfırlanır, periyot süresi devam eder
                new_state[2] = 24.0

            else:
                #  Şut başarısız: top sekiyor (rastgele pozisyona düşüyor)
                new_state[3] += np.random.uniform(-5, 5)
                new_state[4] += np.random.uniform(-5, 5)
                new_state[5] = 5.0
                reward = 0


            # Tüm oyuncuların konumlarını biraz değiştir
            #print(new_state)
            self.update_player_positions(new_state, 6, 16, 2.0)
            self.update_player_positions(new_state, 16, 26, 2.0)

        elif action == 2:  # defend TODO: defen top rakipteyken her zaman yapilan bir sey mi yoksa tek seferlik mi yapilir. 
            ball_owner = int(state[26])
            ball_x = new_state[3]
            ball_y = new_state[4]

            defender_start = 16 
            defender_end = 26 
            # 1. Topa en yakın savunmacıyı bul
            closest_defender_idx = None
            min_distance = float('inf')
            for i in range(defender_start, defender_end, 2):
                dx = new_state[i] - ball_x
                dy = new_state[i+1] - ball_y
                distance = math.sqrt(dx**2 + dy**2)
                if distance < min_distance:
                    min_distance = distance
                    closest_defender_idx = i

            success = random.random() < 0.05  # TODO: bunun istatistigi yok su an arastir

            if success:
                #  Başarılı savunma: top çalınır
                new_state[26] = 3 - ball_owner  # top rakip takıma geçer

                # Top rakip PG oyuncusuna verilir (örnek: takım başlangıç indexi)
                new_owner_start = 6 if new_state[26] == 1 else 16
                new_state[3] = new_state[new_owner_start]
                new_state[4] = new_state[new_owner_start + 1]
                new_state[5] = 5.0

                reward = 0.2

                #  Savunmacılar kutuya yönelir, pozisyon alır
                for i in range(defender_start, defender_end, 2):
                    new_state[i] += np.random.uniform(-2, 2)
                    new_state[i+1] += np.random.uniform(-2, 2)

            else:
                #  Başarısız savunma: topu savunamadılar, oyuncular pozisyon alıyor

                #  En yakın savunmacı topa yaklaşır
                if closest_defender_idx is not None:
                    dx = ball_x - new_state[closest_defender_idx]
                    dy = ball_y - new_state[closest_defender_idx + 1]
                    norm = math.sqrt(dx**2 + dy**2)
                    if norm != 0: #TODO: buraya bir bak
                        dx /= norm
                        dy /= norm
                        new_state[closest_defender_idx] += dx * 2  # baskı
                        new_state[closest_defender_idx + 1] += dy * 2

                # Diğer savunmacılar pota ile top arasına pozisyon alır
                hoop_x = 88 
                hoop_y = 25
                for i in range(defender_start, defender_end, 2):
                    if i != closest_defender_idx:
                        to_ball_x = ball_x - hoop_x
                        to_ball_y = ball_y - hoop_y
                        norm = math.sqrt(to_ball_x**2 + to_ball_y**2)
                        if norm != 0:
                            to_ball_x /= norm
                            to_ball_y /= norm
                            new_state[i] = hoop_x + to_ball_x * 10 + np.random.uniform(-2, 2)
                            new_state[i+1] = hoop_y + to_ball_y * 10 + np.random.uniform(-2, 2)

                reward = 0.0

        elif action == 3:  # dribble
            ball_owner = int(state[26])
            player_start = 6 if ball_owner == 1 else 16
            defender_start = 16 if ball_owner == 1 else 6
            defender_end = 26 if ball_owner == 1 else 16

            ball_x = new_state[3]
            ball_y = new_state[4]

            # Potaya yönelme (hedef koordinat)
            hoop_x = 88 
            hoop_y = 25

            # En yakın savunmacıya olan mesafe
            min_distance = float('inf')
            for i in range(defender_start, defender_end, 2):
                dx = ball_x - new_state[i]
                dy = ball_y - new_state[i+1]
                distance = math.sqrt(dx**2 + dy**2)
                if distance < min_distance:
                    min_distance = distance

            '''
            # Savunmanın gücüne göre başarı ihtimali
            if min_distance < 5: # TODO: buralari sonra duzenle
                success_chance = 0.80
            elif min_distance < 10:
                success_chance = 0.90
            else:
                success_chance = 0.97
            '''

            success = random.random() < 0.99

            if success:
                #  Başarılı dribbling: potaya doğru ilerle
                direction_x = hoop_x - ball_x
                direction_y = hoop_y - ball_y
                norm = math.sqrt(direction_x**2 + direction_y**2)
                if norm != 0:
                    direction_x /= norm
                    direction_y /= norm

                new_state[3] += direction_x * 4  # ilerleme miktarı
                new_state[4] += direction_y * 2
                new_state[5] = 5.0  # top yerde

                reward = 0.001

            else:
                #  Başarısız dribbling: top kaybı
                new_state[26] = 3 - ball_owner  # top rakip takıma geçer

                # Rakip PG'ye top ver
                new_owner_start = 16 
                new_state[3] = new_state[new_owner_start]
                new_state[4] = new_state[new_owner_start + 1]
                new_state[5] = 5.0

                reward = 0

            # 🧭 Topun yönü (dribble yönü)
            dir_x = new_state[3] - state[3]
            dir_y = new_state[4] - state[4]
            norm = math.sqrt(dir_x**2 + dir_y**2)
            if norm != 0:
                dir_x /= norm
                dir_y /= norm

            # 🧍‍♂️ Tüm oyuncular dribbling yönüne doğru küçük hareket yapıyor
            for i in range(6, 26, 2):
                movement_strength = np.random.uniform(0.5, 1.5)  # farklı oyuncular farklı hızda hareket edebilir
                new_state[i] += dir_x * movement_strength + np.random.uniform(-0.5, 0.5)
                new_state[i+1] += dir_y * movement_strength + np.random.uniform(-0.5, 0.5)

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
epsilon = 0.25
epsilon_min = 0.01 
epsilon_decay = 0.995  
batch_size = 32  

# ------------------ 5. Offline Pretraining ------------------ #
# Offline veriyi yükle ve replay buffer'a ekle
#offline_data_path = "with_passes491.json"  # Offline verinizi içeren dosya
offline_data_path = "Last_result_data/21500003_last_result.json"
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
log_dir = "match_logs"
os.makedirs(log_dir, exist_ok=True)
def round_state(state):
    return [round(float(x), 2) for x in state]
episodes = 25  # Online eğitim için epizod sayısı
for episode in range(episodes):
    state = env.reset()
    env.state = torch.FloatTensor(state).to(device)  # state'i doğru cihaza taşı
    total_reward = 0
    done = False
    match_states = []
    while not done:
        valid_actions = env.get_valid_actions(is_home_team=True)
        ball_control = env.state[26]
            # Aksiyon seçimi
        if ball_control == 1:  # Ev sahibi
            #print(env.state)
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = dqn(env.state)
                    action = torch.argmax(q_values).item()
            next_state, reward, done = env.step(action)  # Bu kısımda yeni state alınır
            next_state = next_state.to(device)  # Cihaza taşıma işlemi (önce CPU'ya gerek yok)

            memory.append((env.state, action, reward, next_state, done))  # Memory'e ekle
            env.state = next_state  # State güncellenir (oyun bir sonraki adıma geçer)
            total_reward += reward

        elif ball_control == 2:  # Rakip
            #print(env.state)
            env.state[3] = 47 * 2 - env.state[3]
            for i in range(6, 26, 2):
                env.state[i] = 47 * 2 - env.state[i]  # x pozisyonunu simetrik yap
            #print(env.state)
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = dqn(env.state)
                    action = torch.argmax(q_values).item()
            
            next_state, reward, done = env.step(action)  # Bu kısımda yeni state alınır
            next_state = next_state.to(device)  # Cihaza taşıma işlemi (önce CPU'ya gerek yok)
            next_state[3] = 47 * 2 - next_state[3]
            for i in range(6, 26, 2):
                next_state[i] = 47 * 2 - next_state[i]  # x pozisyonunu simetrik yap
            memory.append((env.state, action, reward, next_state, done))  # Memory'e ekle
            env.state = next_state  # State güncellenir (oyun bir sonraki adıma geçer)
            total_reward += reward
        state_list = env.state.cpu().numpy().tolist()
        match_states.append([round_state(state_list), [], []])


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
    with open(os.path.join(log_dir, f"episode_{episode+1}.json"), "w") as f:json.dump(match_states, f)
    
    # Skor bilgisini al (state'in son iki elemanı)
    score = env.state.cpu().numpy()[-2:]  # CPU'ya taşı ve numpy array'e çevir
    home_score, away_score = score[0], score[1]

    # Epsilon değerini azalt
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}, Score: {home_score:.0f}-{away_score:.0f}")
    #print(env.state)


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

        next_state, reward, done = env.step(action, home=True)  # Aksiyon sonrası yeni state ve ödül alınır
        next_state = next_state.to(device)  # Yeni state'i cihaza taşı

        env.state = next_state  # State güncellenir
        total_reward += reward  # Toplam ödül biriktirilir

    print(f"Test {test + 1}: Toplam Ödül = {total_reward}")

print(q_values)

