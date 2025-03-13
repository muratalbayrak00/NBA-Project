class Config:
    NUM_EPISODES = 10
    MAX_TIMESTEPS = 1000
    GAMMA = 0.99
    LR = 0.0003
    EPSILON = 0.9  # Keşif oranı eklendi
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
