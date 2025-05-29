import torch
import numpy as np


model = torch.load("dqn_model.pth")  # veya torch.load(PATH, map_location=torch.device('cpu'))
model.eval()

# Örnek: ortamdan gelen state vektörün (şu haliyle örnek sayılar)
example_state = np.array([0.1, 0.5, -0.2, 0.0])  # senin ortamına göre değişir

# Torch tensöre çevir
state_tensor = torch.tensor([example_state], dtype=torch.float32)  # batch olarak veriyoruz

# Q-değerlerini tahmin et
with torch.no_grad():
    q_values = model(state_tensor)

print("Q-değerleri:", q_values.numpy())
