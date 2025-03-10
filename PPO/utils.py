import json
import numpy as np

def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    # Data'yı uygun formata dönüştür
    dataset = []
    for entry in data:
        state = np.array(entry[0], dtype=np.float32)
        action = entry[1]
        reward = float(entry[2])
        dataset.append((state, action, reward))

    return dataset
