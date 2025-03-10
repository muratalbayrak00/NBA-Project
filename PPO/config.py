class Config:
    # Eğitim parametreleri
    NUM_EPISODES = 1000
    MAX_TIMESTEPS = 1000
    GAMMA = 0.99  # Discount faktörü
    LR = 0.0003    # Öğrenme oranı
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # Cihaz (GPU/CPU)
