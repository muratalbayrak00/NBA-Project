import pandas as pd
import csv
import json
from pathlib import Path
import numpy as np
import Filter_Code.find_passes as find_passes

# CSV dosyasını oku
df = pd.read_csv('Row_data/shots_fixed.csv')

#game_ids = {'21500001', "21500002", "21500491"}

# Benzersiz GAME_ID'leri al
game_ids = df['GAME_ID'].unique()

for game_id in game_ids:
    find_passes.update_json_with_actions(game_id)