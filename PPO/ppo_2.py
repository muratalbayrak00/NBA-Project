import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PPOAgent:
    def __init__(self, state_dim, action_dim, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Model ve optimizer'ı tanımla
        self.policy = ActorCritic(state_dim, action_dim).to(config.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.LR)
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.config.device)
        action_probs, _ = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).to(self.config.device)
        next_state = torch.FloatTensor(next_state).to(self.config.device)
        action = torch.tensor(action).to(self.config.device)
        reward = torch.tensor(reward).to(self.config.device)

        # Actor-Critic güncellemeleri
        action_probs, value = self.policy(state)
        _, next_value = self.policy(next_state)

        # Advantage hesapla
        advantage = reward + (1 - done) * self.config.GAMMA * next_value - value

        # PPO objective fonksiyonunu hesapla
        action_log_prob = torch.log(action_probs[action])
        loss = -action_log_prob * advantage + 0.5 * advantage**2

        # Optimize
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value
    
