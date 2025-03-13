import numpy as np
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
import logging


logging.basicConfig(
    filename="training_log.txt",  # Log dosyasının adı
    level=logging.ERROR,           # Sadece ERROR seviyesindeki loglar kaydedilecek
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# ------------------
# Ortam ve State İşleme
# ------------------
#home_player_ids = {2547, 2548, 2405, 201609, 204020, 2736, 2365, 101123, 2757, 1626159, 202355, 2617}
#visitor_player_ids = {101109, 101111, 200826, 203939, 101114, 201580, 203098, 202379, 202083, 202718, 2585, 2734, 1717}

home_player_ids = {2547, 2548, 2405, 201609, 204020}
visitor_player_ids = {101109, 101111, 200826, 203939}

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
        #with open(self.json_file_path, "r") as file:
            #self.data = json.load(file)
    
        #self.events = self.data["events"]
        #self.event_idx = 0
        #self.moment_idx = 1
        self.done = False
        self.state = [67.918,39.34429,84.41229,14.72567,64.50734,22.22941,90.77177,35.9453,86.46641,47.68162,78.56783,28.44641,83.9457,33.11776,90.04088,32.31945,85.39303,16.24225,72.85798,21.3308,84.14827,14.00846,1.34399,700,2736, 0, 0]
        return self.state
    
    
    def step(self, action):
        """
        Bir aksiyonu uygular ve yeni state, ödül ve oyunun bitip bitmediğini döndürür.
        """
        reward = 0
        result = None

        try:
            # Aksiyon türüne göre işlemler
            if action == "shoot":
                # Şutun başarı oranı, saha pozisyonuna bağlı olabilir
                ball_position = (self.state[20],self.state[21])  # Topun mevcut pozisyonu
                distance_to_basket = np.linalg.norm(ball_position - np.array(self.home_basket_coords))
                success_rate = max(0.1, 0.9 - 0.02 * distance_to_basket)  # Daha uzak mesafede daha düşük başarı
                if random.random() < success_rate:
                    reward = 10  # Başarılı şut
                    result = "shot_made"
                    
                else:
                    reward = -1  # Başarısız şut
                    result = "shot_missed"

            elif action == "pass":
                # Pasın başarı oranı, top tutan oyuncu ve yakınındaki oyuncuların pozisyonlarına bağlı olabilir
                ball_holder = int(self.state[24])  # Topu tutan oyuncu
                if ball_holder != 0:  # Eğer top bir oyuncuda ise
                    success_rate = 0.9  # Sabit başarı oranı, daha detaylı pozisyon analiziyle geliştirilebilir
                    if random.random() < success_rate:
                        reward = 3  # Başarılı pas
                        result = "pass_success"
                    else:
                        reward = -1  # Top kaybı
                        result = "turnover"
                else:
                    reward = -1  # Top kaybı
                    result = "turnover"

            elif action == "dribble":
                result = "dribble"
            
            elif action == "defend":
                # Savunma aksiyonları
                ball_holder = int(self.state[24])       
                steal_chance = 0.3  # Top çalma olasılığı
                if random.random() < steal_chance:
                    reward = 5  # Başarılı top çalma
                    result = "steal"
                else:
                    reward = 0  # Başarısız savunma
                    result = "defend_fail"
            print(result)
            # State güncelle
            #print(self.done)
            self.state, self.done = self.update_next_state(result)
            #(self.done)
            #print(f"State: {state}")
        except Exception as e:
            logging.error(f"Error in step function: {e}")
            #self.done = True

        return self.state, reward, self.done#, result

    def update_next_state(self, result):
        """
        Bu fonksiyon, verilen sonuç (result) temelinde state'i günceller.
        """
        try:
            # Mevcut durumu alın
            new_state = self.state.copy()
            
            # Sonuca göre durumu güncelle
            if result == "shot_made":
                # Şut isabetli, top rakip bir oyuncuya geçer
                ball_holder = random.choice(list(visitor_player_ids))  # Rastgele bir rakip oyuncu seç
                new_state[24] = ball_holder  # Topu bu oyuncuya ata

                # Ev sahibi oyuncular kendi sahasında olsun (0'dan 5'e kadar)
                for i in range(5):
                    new_state[i * 2] = random.uniform(0, 47)  # X koordinatı (kendi sahasında)
                    new_state[i * 2 + 1] = random.uniform(0, 50)  # Y koordinatı

                # Rakip takım oyuncuları kendi sahasında olsun (5'ten 10'a kadar)
                visitor_positions = []  # Rakip oyuncuların pozisyonlarını saklamak için liste
                for i in range(5, 10):
                    x = random.uniform(47, 94)  # X koordinatı (rakip sahasında)
                    y = random.uniform(0, 50)   # Y koordinatı
                    new_state[i * 2] = x
                    new_state[i * 2 + 1] = y
                    visitor_positions.append((x, y))  # Rakip oyuncunun pozisyonunu kaydet

                # Topun konumunu rakip oyunculardan birine yakın olacak şekilde ayarla
                selected_player_pos = random.choice(visitor_positions)  # Rastgele bir rakip oyuncu seç
                new_state[20] = selected_player_pos[0] + random.uniform(-1, 1)  # Topun X koordinatı
                new_state[21] = selected_player_pos[1] + random.uniform(-1, 1)  # Topun Y koordinatı
                new_state[22] = random.uniform(0, 10)  # Topun Z koordinatı (örneğin, maksimum 10 birim yükseklik)

                new_state[25] += 2

            elif result == "shot_missed":
                all_player_ids = list(home_player_ids) + list(visitor_player_ids)
                ball_holder = random.choice(all_player_ids)  # Rastgele bir oyuncu seç
                new_state[24] = ball_holder  # Topu bu oyuncuya ata

                # Tüm oyuncuları rastgele biraz hareket ettir (200 ms'lik hareket)
                all_player_positions = []
                for i in range(10):  # Toplam 10 oyuncu (5 ev sahibi, 5 rakip)
                    current_x = new_state[i * 2]
                    current_y = new_state[i * 2 + 1]
                    new_state[i * 2] = max(0, min(94, current_x + random.uniform(-1, 1)))  # X koordinatını güncelle
                    new_state[i * 2 + 1] = max(0, min(50, current_y + random.uniform(-1, 1)))  # Y koordinatını güncelle
                    all_player_positions.append((new_state[i * 2],new_state[i * 2 + 1]))

                selected_player_pos = random.choice(all_player_positions)
                new_state[20] = selected_player_pos[0] + random.uniform(-1, 1)  # Topun X koordinatı
                new_state[21] = selected_player_pos[1] + random.uniform(-1, 1)  # Topun Y koordinatı
                new_state[22] = random.uniform(0, 10)  # Topun Z koordinatı (örneğin, maksimum 10 birim yükseklik)


            elif result == "pass_success":
                # Başarılı pas, top ev sahibi takım oyuncusuna geçer
                ball_holder = random.choice(list(home_player_ids))  # Rastgele bir ev sahibi oyuncu seç
                new_state[24] = ball_holder  # Topu bu oyuncuya ata

                # Tüm oyuncuları rastgele biraz hareket ettir (200 ms'lik hareket)
                for i in range(10):  # Toplam 10 oyuncu (5 ev sahibi, 5 rakip)
                    current_x = new_state[i * 2]
                    current_y = new_state[i * 2 + 1]
                    new_state[i * 2] = max(0, min(94, current_x + random.uniform(-1, 1)))  # X koordinatını güncelle
                    new_state[i * 2 + 1] = max(0, min(50, current_y + random.uniform(-1, 1)))  # Y koordinatını güncelle
                

                ball_x = new_state[random.uniform(0, 5) * 2]
                ball_y = new_state[random.uniform(0, 5) * 2 + 1]
                new_state[20] = ball_x + random.uniform(-2, 2)  # Topun X koordinatı
                new_state[21] = ball_y + random.uniform(-2, 2)  # Topun Y koordinatı
                new_state[22] = random.uniform(0, 10)  # Topun Z koordinatı (örneğin, maksimum 10 birim)

            elif result == "turnover":
                ball_holder = random.choice(list(visitor_player_ids))  # Rastgele bir rakip oyuncu seç
                new_state[24] = ball_holder  # Topu bu oyuncuya ata

                # Tüm oyuncuları rastgele biraz hareket ettir (200 ms'lik hareket)
                for i in range(10):  # Toplam 10 oyuncu (5 ev sahibi, 5 rakip)
                    current_x = new_state[i * 2]
                    current_y = new_state[i * 2 + 1]
                    new_state[i * 2] = max(0, min(94, current_x + random.uniform(-1, 1)))  # X koordinatını güncelle
                    new_state[i * 2 + 1] = max(0, min(50, current_y + random.uniform(-1, 1)))  # Y koordinatını güncelle
                

                ball_x = new_state[random.uniform(5, 10) * 2]
                ball_y = new_state[random.uniform(5, 10) * 2 + 1]
                new_state[20] = ball_x + random.uniform(-2, 2)  # Topun X koordinatı
                new_state[21] = ball_y + random.uniform(-2, 2)  # Topun Y koordinatı
                new_state[22] = random.uniform(0, 10)  # Topun Z koordinatı (örneğin, maksimum 10 birim)
                
            elif result == "steal":
                # Başarılı pas, top ev sahibi takım oyuncusuna geçer
                ball_holder = random.choice(list(home_player_ids))  # Rastgele bir ev sahibi oyuncu seç
                new_state[-1] = ball_holder  # Topu bu oyuncuya ata

                # Tüm oyuncuları rastgele biraz hareket ettir (200 ms'lik hareket)
                for i in range(10):  # Toplam 10 oyuncu (5 ev sahibi, 5 rakip)
                    current_x = new_state[i * 2]
                    current_y = new_state[i * 2 + 1]
                    new_state[i * 2] = max(0, min(94, current_x + random.uniform(-1, 1)))  # X koordinatını güncelle
                    new_state[i * 2 + 1] = max(0, min(50, current_y + random.uniform(-1, 1)))  # Y koordinatını güncelle

                ball_x = new_state[20]  # Topun X koordinatı
                ball_y = new_state[21]  # Topun Y koordinatı

                # En yakın ev sahibi oyuncuyu bul
                min_distance = float('inf')  # Başlangıçta mesafe çok büyük
                closest_player_index = -1

                for i in range(5):  # 5 ev sahibi oyuncu
                    player_x = new_state[i * 2]  # Oyuncunun X koordinatı
                    player_y = new_state[i * 2 + 1]  # Oyuncunun Y koordinatı

                    # Mesafeyi hesapla (Euclidean distance)
                    distance = np.sqrt((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2)

                    if distance < min_distance:  # En küçük mesafeyi bul
                        min_distance = distance
                        closest_player_index = i

                # Topun konumunu en yakın ev sahibi oyuncuya yakın olacak şekilde ayarla
                new_state[20] = new_state[closest_player_index * 2] + random.uniform(-2, 2)  # Topun X koordinatını güncelle
                new_state[21] = new_state[closest_player_index * 2 + 1] + random.uniform(-2, 2)  # Topun Y koordinatını güncelle
                new_state[22] = random.uniform(0, 10)  # Topun Z koordinatını (örneğin, maksimum 10 birim)
                

            elif result == "defend_fail":

                # Tüm oyuncuları rastgele biraz hareket ettir (200 ms'lik hareket)
                for i in range(10):  # Toplam 10 oyuncu (5 ev sahibi, 5 rakip)
                    current_x = new_state[i * 2]
                    current_y = new_state[i * 2 + 1]
                    new_state[i * 2] = max(0, min(94, current_x + random.uniform(-1, 1)))  # X koordinatını güncelle
                    new_state[i * 2 + 1] = max(0, min(50, current_y + random.uniform(-1, 1)))  # Y koordinatını güncelle

                new_state[20] = new_state[20] + random.uniform(-1, 1) 
                new_state[21] = new_state[21] + random.uniform(-1, 1) 
                new_state[22] = new_state[22] + random.uniform(-1, 1) 
            
            elif result == "dribble":

                for i in range(10):  # Toplam 10 oyuncu (5 ev sahibi, 5 rakip)
                    current_x = new_state[i * 2]
                    current_y = new_state[i * 2 + 1]
                    new_state[i * 2] = max(0, min(94, current_x + random.uniform(-1, 1)))  # X koordinatını güncelle
                    new_state[i * 2 + 1] = max(0, min(50, current_y + random.uniform(-1, 1)))  # Y koordinatını güncelle

                new_state[20] = new_state[20] + random.uniform(-1, 1) 
                new_state[21] = new_state[21] + random.uniform(-1, 1) 
                new_state[22] = new_state[22] + random.uniform(-1, 1) 

            elif result == "idle":
                for i in range(10):  # Toplam 10 oyuncu (5 ev sahibi, 5 rakip)
                    current_x = new_state[i * 2]
                    current_y = new_state[i * 2 + 1]
                    new_state[i * 2] = max(0, min(94, current_x + random.uniform(-1, 1)))  # X koordinatını güncelle
                    new_state[i * 2 + 1] = max(0, min(50, current_y + random.uniform(-1, 1)))  # Y koordinatını güncelle

                new_state[20] = new_state[20] + random.uniform(-1, 1) 
                new_state[21] = new_state[21] + random.uniform(-1, 1) 
                new_state[22] = new_state[22] + random.uniform(-1, 1) 
            print(new_state[23])
            new_state[23] -= 0.2 
            # Oyun bitişini kontrol et
            if new_state[23] <= 0:  # Kalan süre 0 veya daha küçükse oyun biter
                self.done = True
                print("Game Over! Time's up.")
            else:
                self.done = False
            print(3000)
            print(self.done)
            return new_state, self.done

        except Exception as e:
            logging.error(f"Error in update_next_state function: {e}")
            return self.state, True  # Hata durumunda oyunu bitir

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
actions = ["pass", "shoot", "dribble", "defend", "idle"]


# Hyperparametreler
state_dim = 27  # State'in boyutu
action_dim = len(actions)
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
replay_buffer = []



# ------------------
# Eğitim Döngüsü
# ------------------
def train_dqn(batch_size=32):
    if len(replay_buffer) < batch_size:
        return

    batch = random.sample(replay_buffer, batch_size)
    #print(f"Replay Buffer Entry: {batch}")  # Bu satırı ekleyin
    #print("_________________________________________________________________________")

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


env = BasketballEnvironment("../Data/find_action/match_data.json", [5.37, 24.7], [88, 24.7])


for episode in range(3):
    state = env.reset()
    total_reward = 0

    while True:
        #print("while")
        if random.random() < epsilon:
            print(4001)
            action_idx = random.randint(0, action_dim - 1)
        else:
            print(4002)
            with torch.no_grad():
                q_values = model(torch.tensor(state, dtype=torch.float32))
                action_idx = torch.argmax(q_values).item()

        action = actions[action_idx]
        next_state, reward, done = env.step(action)
        print(done)
        #replay_buffer.clear
        max_size = 10000
        if len(replay_buffer) > max_size:
            replay_buffer = replay_buffer[-max_size:]  # Eski verileri atar, son `max_size` kadarını tutar.

        # Replay buffer'a ekleme sırasında None kontrolü
        if next_state is not None:
            print(4003)
            replay_buffer.append((state, action_idx, reward, next_state, done))
            if len(replay_buffer) > 10000:
                print(4004)
                replay_buffer.pop(0)
        else:
            print(4005)
            logging.warning(f"Next state is None at Event {env.event_idx}, Moment {env.moment_idx}")

        train_dqn()
        #print(done)

        #state = next_state if next_state is not None else state
        total_reward += reward
        if done:
            print("break")
            break

    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")