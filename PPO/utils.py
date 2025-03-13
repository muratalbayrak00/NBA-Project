import json
import numpy as np

def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    action_mapping = {
        "defend_fail": 0,
        "succesfull_defend": 1,
        "idle_defend": 2,
        "dribble": 3,
        "succesfull_shot": 4,
        "missed_shot": 5,
        "succesfull_pass": 6,
        "missed_pass": 7,
        "idle_pass": 8,
        "idle": 9
    }

    dataset = []
    for entry in data:
        state = np.array(entry[0], dtype=np.float32)
        action = action_mapping[entry[1][0]]
        reward = float(entry[2])
        dataset.append((state, action, reward))

    return dataset
