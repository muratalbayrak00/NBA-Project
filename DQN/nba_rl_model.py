import numpy as np
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
import logging

# ------------------
# Loglama Ayarları
# ------------------
logging.basicConfig(
    filename="training_log.txt",  # Log dosyasının adı
    level=logging.ERROR,           # Sadece ERROR seviyesindeki loglar kaydedilecek
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------
# Ortam ve State İşleme
# ------------------

class BasketballEnvironment:
    def __init__(self, json_file_path, home_basket_coords, visitor_basket_coords):
        self.json_file_path = json_file_path
        self.home_basket_coords = home_basket_coords
        self.visitor_basket_coords = visitor_basket_coords
        self.reset()

    def reset(self):
        """
        Oyunu sıfırlar ve başlangıç state'ini döndürür.
        """
        with open(self.json_file_path, "r") as file:
            self.data = json.load(file)
        
        self.events = self.data["events"]
        self.event_idx = 0
        self.moment_idx = 1
        self.state, self.done = self.process_moment()
        return self.state
    
    def identify_ball_holder(self, current_moment):
        ball_position = current_moment[5][0][2:4]
        player = np.array([
                [player[1], player[2], player[3]] for player in current_moment[5]
            ])
        player = np.delete(player, 0, axis=0)  # Topu çıkar
        min_distance = 90
        min_id = 0
        for pos in player:
            player_position = pos[1:]
            distance = np.linalg.norm(player_position - ball_position) # euclidian 
            #print(distance)
            if (distance < min_distance):
                min_distance = distance
                min_id = pos[0]
                #print( min_distance)
                #print(min_id)
        min_id = int(min_id)
        if (min_distance < 2):
            return min_id
        else:
            return 0
        
    def process_moment(self):
        """
        JSON verisinden bir moment işleyerek state'i oluşturur.
        """
        try:
            event = self.events[self.event_idx]
            moments = event["moments"]

            # Bir önceki ve mevcut moment
            previous_moment = moments[self.moment_idx - 1]
            current_moment = moments[self.moment_idx]

            # Oyuncu konumlarını ve top konumunu al
            player_positions = np.array([
                [player[2], player[3]] for player in current_moment[5]
            ])
            player_positions = np.delete(player_positions, 0, axis=0)  # Topu çıkar
            #print(player_positions)
            ball_position = current_moment[5][0][2:]  # Topun koordinatları
            time_remaining = current_moment[3]  # Zaman bilgisi

             # Topu tutan oyuncuyu bul
            ball_holder = self.identify_ball_holder(current_moment)
            #print(f"Ball holder: {ball_holder}")
            # State'i oluştur
            state = np.concatenate((player_positions.flatten(), ball_position, [time_remaining, ball_holder])) # TODO: state'lerimize skor eklenmeli 
            np.set_printoptions(suppress=True)
            #print(f"State: {state}")

            # Eğer state beklenen boyutta değilse hata ver
            if len(state) != 25:
                logging.error(f"Unexpected state size in Event {self.event_idx}, Moment {self.moment_idx}: {len(state)}")
                print("burası patlıyor.")
                self.moment_idx += 1
                if self.moment_idx >= len(moments):
                    self.event_idx += 1
                    self.moment_idx = 1
                    self.process_moment()
                    if self.event_idx >= len(self.events):
                        print("----------------------------------------------------------------------")
                        print(self.event_idx)
                        print(self.events)
                        return None, True  # Tüm eventler bitti

            # Bir sonraki momenti işlemek için ilerle
            self.moment_idx += 1
            if self.moment_idx >= len(moments):
                self.event_idx += 1
                self.moment_idx = 1
                if self.event_idx >= len(self.events):
                    print("----------------------------------------------------------------------")
                    print(self.event_idx)
                    print(self.events)
                    return None, True  # Tüm eventler bitti

            return state, False

        except Exception as e:
            logging.error(f"Error processing moment: {e}")
            return None, True
    
    def step(self, action):
        """
        Bir aksiyonu uygular ve yeni state, ödül ve oyunun bitip bitmediğini döndürür
        """
        reward = 0
        result = None

        try:
            # Aksiyon türüne göre işlemler
            if action == "shoot":
                # Şutun başarı oranı, saha pozisyonuna bağlı olabilir
                ball_position = self.state[-4:-2]  # TODO: burayi degistir topun x y z olarak almali burada 2 tane aliyor [-5,-3] olmali????
                distance_to_basket = np.linalg.norm(ball_position - np.array(self.home_basket_coords))
                success_rate = max(0.1, 0.9 - 0.02 * distance_to_basket)  # Daha uzak mesafede daha düşük başarı
                if random.random() < success_rate:
                    reward = 15  # Başarılı şut
                    result = "shot_made"
                    print(result)
                else:
                    reward = -2  # Başarısız şut
                    result = "shot_missed"
                    print(result)

            elif action == "pass":
                # Pasın başarı oranı, top tutan oyuncu ve yakınındaki oyuncuların pozisyonlarına bağlı olabilir
                ball_holder = int(self.state[-1])  # Topu tutan oyuncu
                if ball_holder != 0:  # TODO: passi kime atiyor rakip oyuncu mu takim arkadasi mi 
                    success_rate = 0.9  # Sabit başarı oranı, daha detaylı pozisyon analiziyle geliştirilebilir
                    if random.random() < success_rate:
                        reward = 3  # Başarılı pas
                        result = "pass_success"
                        print(result)
                    else:
                        reward = -1  # Top kaybı
                        result = "turnover"
                        print(result)

            elif action in ["move_up", "move_down", "move_left", "move_right"]:
                # Oyuncunun hareketi
                movement = {
                    "move_up": [0, 1],
                    "move_down": [0, -1],
                    "move_left": [-1, 0],
                    "move_right": [1, 0]
                }
                player_positions = self.state[:-4].reshape(-1, 2)
                ball_holder = int(self.state[-1])
                if ball_holder > 0:  # Eğer top bir oyuncudaysa
                    player_positions[ball_holder - 1] += movement[action]  # Top tutan oyuncunun pozisyonunu değiştir
                    reward = 1  # Hareket için küçük bir ödül
                    result = "move_success"
                    print(result)

            elif action == "defend":
                # Savunma aksiyonları
                ball_holder = int(self.state[-1])
                if ball_holder > 0:  # Eğer top bir oyuncudaysa
                    steal_chance = 0.3  # Top çalma olasılığı
                    if random.random() < steal_chance:
                        reward = 5  # Başarılı top çalma
                        self.state[-1] = 0  # Top artık serbest
                        result = "steal"
                        print(result)
                    else:
                        reward = -1  # Başarısız savunma
                        result = "defend_fail"
                        print(result)

            # State güncelle
            self.state, self.done = self.process_moment()
            print(f"State: {state}")
        except Exception as e:
            logging.error(f"Error in step function: {e}")
            #self.done = True

        return self.state, reward, self.done#, result


# ------------------
# DQN Modeli
# ------------------

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

# Aksiyonlar
actions = [
    "pass", "shoot", "move_up", "move_down",
    "move_left", "move_right", "dribble", "defend", "idle"
]

# Hyperparametreler
state_dim = 25  # State'in boyutu
action_dim = len(actions)
model = DQN(state_dim, action_dim)
target_model = DQN(state_dim, action_dim)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

epsilon = 1.0 # rastgelelilik 
epsilon_decay = 0.995 # epsilon azalma orani 
epsilon_min = 0.01 #  min epsilon
gamma = 0.99 # gelecekteki odullere buyuk agirlik vermesi
replay_buffer = [] 

# ------------------
# Eğitim Döngüsü
# ------------------

def train_dqn(batch_size=32):
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

# Eğitim Başlangıcı
env = BasketballEnvironment("../Data/filtered_data/fdni0021500491.json", [5.37, 24.7], [88, 24.7])

for episode in range(200):
    state = env.reset()
    total_reward = 0

    while True:
        #print("while")
        if random.random() < epsilon:
            action_idx = random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                q_values = model(torch.tensor(state, dtype=torch.float32))
                action_idx = torch.argmax(q_values).item()

        action = actions[action_idx]
        next_state, reward, done = env.step(action)

        # Replay buffer'a ekleme sırasında None kontrolü
        if next_state is not None:
            replay_buffer.append((state, action_idx, reward, next_state, done))
            if len(replay_buffer) > 10000:
                replay_buffer.pop(0)
        else:
            logging.warning(f"Next state is None at Event {env.event_idx}, Moment {env.moment_idx}")

        train_dqn()

        state = next_state if next_state is not None else state
        total_reward += reward
        if done:
            print("break")
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Modeli kaydetme (eğitim tamamlandıktan sonra)
torch.save(model.state_dict(), "dqn_model.pth")
#print("Model kaydedildi.")
# Modeli yükleme
model.load_state_dict(torch.load("dqn_model.pth", weights_only=True))
model.eval()  # Modeli değerlendirme moduna alır
#print("Model yüklendi ve değerlendiriliyor.")
# Modelin bir state üzerinde aksiyon tahmin etmesi
state = env.reset()  # Örnek bir başlangıç durumu
with torch.no_grad():
    q_values = model(torch.tensor(state, dtype=torch.float32))
    action_idx = torch.argmax(q_values).item()

#print(f"Modelin verdiği aksiyon: {action_idx}")

