import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import json

with open("/find_action/match_data.json", "r") as f:
    data = json.load(f)

# Datayi ayristir.
states = [item[0] for item in data]
actions = [item[1][0] for item in data]  # Action'ları string'den index'e çevir
rewards = [item[2] for item in data]

# Action'ları index'e çevirir
action_map = {"idle": 0, "dribble": 1, "pass": 2, "shoot": 3, "defend": 4}
actions = [action_map[action] for action in actions]

# tensor'a çevirme
states = torch.FloatTensor(states)
actions = torch.LongTensor(actions)
rewards = torch.FloatTensor(rewards)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim): # state_dim => state boyutu (25)
        super(ActorCritic, self).__init__()
        # Ortak katmanlar
        self.shared_layers = nn.Sequential( # burasi stateleri isler.
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # aktör (Politika) ag
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        # critic ag
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # state'in değerini temsil eden tek bir skaler degerdir.
        )
    # burasi statei alır ortak katmanlardan geçirir, aktör ve kritik ağlarını kullanarak aksiyon olasılıklarını ve state değerini hesaplar
    def forward(self, state):
        shared = self.shared_layers(state)
        action_logits = self.actor(shared) # logits her bir değer, bir aksiyonun "ham skorunu" temsil eder.
        action_logits = torch.clamp(action_logits, min=-50, max=50)  # logits'i sınırla, olasilik hesaplamasinin kararli olmasi icin gerekli
        action_probs = nn.Softmax(dim=-1)(action_logits) # olasiliga donusturur burada 
        value = self.critic(shared)
        return action_probs, value

class PPO: # lr = learning rate, gamma = discount factor, epsilon = güncelleme oranını çok büyük değişimlerden korumak için katsayi
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr) # parametrelerin (ağırlıkların) güncellenmesi işlemini yapar
        self.MseLoss = nn.MSELoss() # modelin tahminleri ile gerçek değerler arasındaki farkın karelerinin ortalamasını alır

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample() # rastgele secim yapar. 
        log_prob = dist.log_prob(action) # secilen aksiyonun logritmik olasiligi alinir. 
        return action.item(), log_prob.item()

    def update(self, states, actions, log_probs_old, rewards, next_states, dones):
        # odulleri gamma ile güncelle
        discounted_rewards = []
        total_reward = 0

        for reward, done in zip(reversed(rewards), reversed(dones)):
            total_reward = reward + (1 - done) * self.gamma * total_reward
            discounted_rewards.insert(0, total_reward)
        
        # Tensor'a dönüştür
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        log_probs_old = torch.FloatTensor(np.array(log_probs_old))
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        
        # odul normalizasyon ( cok buyuk ve cok kucuk reward degerlerinin egitim dengesizligini korur)
        std = discounted_rewards.std()
        if std == 0 or torch.isnan(std):
            discounted_rewards = discounted_rewards - discounted_rewards.mean()
        else:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (std + 1e-7)
        
        #3 epoch boyunca güncelle
        for _ in range(3): # buradaki 3 u degistirebiliriz. cok olursa overfitting olur az olmasida yeterli degil. 
            action_probs, values = self.policy(states)
            dist = Categorical(action_probs)
            log_probs_new = dist.log_prob(actions)
            
            # Advantage hesapla
            advantages = discounted_rewards - values.squeeze()
            
            # Oran (ratio) ve kayıp fonksiyonu
            ratios = torch.exp(log_probs_new - log_probs_old) # Yeni ve eski politika arasındaki değişim oranı
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Kritik kaybı
            critic_loss = self.MseLoss(values.squeeze(), discounted_rewards)
            
            # Toplam kayıp
            total_loss = actor_loss + 0.5 * critic_loss # (eğer critic_loss büyük olursa Actor’un öğrenmesini baskılayabilir.)
            
            # Geri yayılım
            self.optimizer.zero_grad() # önceki iterasyonlardan kalan gradyanlar sifirlanir. 
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)  # gradyan kırpma ( Aşırı güncellemeyi onlemek icin yapiyoruz.)
            self.optimizer.step()

# Hiperparametreler
state_dim = 25  # 10 oyuncu * 2 koordinat + 3 top koordinatı + 1 süre + 1 oyuncu ID
action_dim = 5   # 5 ayrık aksiyon
ppo_agent = PPO(state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2)

# Eğitim döngüsü
for episode in range(1000):
    episode_reward = 0
    for i in range(len(states) - 1):
        state = states[i]
        action, log_prob = ppo_agent.select_action(state)
        next_state = states[i + 1]
        reward = rewards[i]
        done = (i == len(states) - 2)
        ppo_agent.update([state], [action], [log_prob], [reward], [next_state], [done])
        episode_reward += reward
    
    # Episode başına ödülü yazdır
    print(f"Episode {episode + 1}, Total Reward: {episode_reward}")

# Modeli kaydet
torch.save(ppo_agent.policy.state_dict(), "ppo_model.pth")

# JSON dosyası okunur:
# Veri ayrıştırılır:
# Aksiyonlar indekslere dönüştürülür:
# Actor-Critic sinir ağı modeli tanımlanır:
# PPO (Proximal Policy Optimization) algoritması tanımlanır:
# Aksiyon seçme fonksiyonu (select_action) tanımlanır:
# Ajanın güncellenmesi için update fonksiyonu tanımlanır:
# Eğitim döngüsü başlatılır (1000 episode):