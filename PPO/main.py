import json
import numpy as np
import torch
from ppo_1 import PPOAgent
from utils import load_data

# Konfigürasyonları yükle
from config import Config

def update_state(state, action):
    """
    State güncelleme fonksiyonu:
    - Süreyi 0.2 saniye azaltır.
    - Pas, şut veya top çalma gibi aksiyonlarda top sahibi değişir.
    - Dribble veya pas aksiyonlarında top hareket ettirilir.
    
    oncelikle su aktionlari duzeltmek gerekiyor 
    mantikli actionlar => pas, shot, dribble, defend;

    
    action_mapping = {
        "pass": 0,
        "shot": 1,
        "dribble": 2,
        "defend": 3,
    }

    pass    olursa => top sahipligini bir pasi alan oyuncuya ata, topun konumunu yeni oyuncunun x y konumu olarak ayarla z konumunu bilmiyorum bunu dusunmek lazim.
    shot    olursa => topun konumu potaya dogru gidecek, scor artacak oyuncunun basket attigi yere gore, top sahipligi degisecek 0 mi olur bilmem ( belki direk konumu cemberin konumu olarak ayarlariz topu hareket ettirmeden)
    dribble olursa => top sahipligi degismeyecek, top sahip oyuncu ile topun konumu yakin olacak. 
    defend  olursa => oyuncularin konumlari topa dogru (x dogrultusu) yonelmeli, top sahipligi egitilen takima gecene kadar defend devam etmeli. yada sure bitene kadar.

    bunlari yaparken ev sahibi ve rakip takimi belirleemk gerekiyor. ilk 10 veri ev sahibi kalan rakip gibi vs 
    pass durumunda top sahipligini kontrol edebilmek icin oyuncu idlerini tuttugumuz bir array gerekli. bunu veriden almak gerekiyor
    shot icin 2 lik 3 luk ve 1 lik icin tanimlamalar yapilmali bolgeler ayirt edilmeli.
    Döndürür:
    Güncellenmiş state
    """
    new_state = state.copy()
    
    # Süreyi 0.2 saniye azalt
    new_state[24] = max(0, new_state[24] - 0.2)
    
    # Oyuncu koordinatları (ilk 20 değer)
    player_positions = np.array(new_state[:20]).reshape(10, 2)
    
    # Topun konumu (21, 22, 23. indeksler)
    ball_position = np.array(new_state[21:24])
    
    # Topun sahibi (son eleman)
    ball_owner = int(new_state[24])
    
    if action == 3:  # dribble
        """ Dribbling yapan oyuncu topu 1-2 birim hareket ettirir. """
        move_vector = np.random.uniform(-2, 2, size=2)  # Rastgele hareket
        player_positions[ball_owner] += move_vector  # Top sahibi oyuncu hareket eder
        ball_position[:2] += move_vector  # Top da oyuncuyla birlikte hareket eder
    
    elif action == 6:  # succesfull_pass
        """ Pas başarılıysa, yeni oyuncuya geçer ve topun konumu değişir. """
        new_owner = np.random.choice([i for i in range(10) if i != ball_owner])
        ball_position[:2] = player_positions[new_owner]  # Yeni oyuncunun konumuna geç
        new_state[24] = new_owner  # Yeni sahipliği güncelle
    
    elif action == 7:  # missed_pass
        """ Pas başarısızsa, top rastgele bir noktaya düşer. """
        ball_position[:2] += np.random.uniform(-5, 5, size=2)  # Rastgele bir noktaya düş
        new_state[24] = -1  # Sahipsiz hale gelir
    
    elif action == 4:  # succesfull_shot
        """ Başarılı şut ise, top kaleye gider ve skor değişir. """
        ball_position[:2] = [100, 50]  # Örneğin, kale noktası
        new_state[24] = -1  # Top artık kimseye ait değil
    
    elif action == 5:  # missed_shot
        """ Kaçan şut ise, top sahada rastgele bir noktaya düşer. """
        ball_position[:2] = np.random.uniform([0, 0], [100, 50])  # Saha içinde rastgele bir konum
        new_state[24] = -1  # Top boşa çıktı
    
    # Yeni güncellenmiş state değerlerini geri yaz
    new_state[:20] = player_positions.flatten().tolist()
    new_state[21:24] = ball_position.tolist()
    
    return new_state

def main():
    # Veriyi JSON dosyasından yükle
    dataset = load_data('basketball_data.json')  # JSON verisini yükle

    # PPO Agent'ını başlat
    agent = PPOAgent(state_dim=25, action_dim=10, config=Config)  # State ve Action boyutlarını belirt

    # Eğitim döngüsüne başla
    for episode in range(Config.NUM_EPISODES):
        state = dataset[0][0]  # İlk durumu al
        done = False
        episode_reward = 0

        for t in range(Config.MAX_TIMESTEPS):
            action = agent.select_action(state)
            next_state = update_state(state, action)  # State güncellendi
            reward = dataset[t + 1][2] if t + 1 < len(dataset) else 0
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward

            if done:
                break
        
        print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

if __name__ == "__main__":
    main()
