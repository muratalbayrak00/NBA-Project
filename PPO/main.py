import json
import numpy as np
import torch
from ppo import PPOAgent
from utils import load_data

# Konfigürasyonları yükle
from config import Config

def main():
    # Veriyi JSON dosyasından yükle
    dataset = load_data('basketball_data.json')  # JSON verisini yükle

    # PPO Agent'ını başlat
    agent = PPOAgent(state_dim=25, action_dim=5, config=Config)  # State ve Action boyutlarını belirt

    # Eğitim döngüsüne başla
    for episode in range(Config.NUM_EPISODES):
        state = dataset[0][0]  # İlk durumu al
        done = False
        episode_reward = 0

        for t in range(Config.MAX_TIMESTEPS):
            action = agent.select_action(state)
            next_state, reward, done = dataset[t + 1]  # Bir sonraki durumu ve ödülü al

            # PPO algoritmasını güncelle
            agent.update(state, action, reward, next_state, done)

            state = next_state  # Durumu güncelle
            episode_reward += reward  # Toplam ödülü güncelle

            if done:
                break
        
        print(f"Episode {episode + 1}: Total Reward: {episode_reward}")

if __name__ == "__main__":
    main()
