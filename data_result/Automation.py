import pandas as pd
import csv
import json
from pathlib import Path
import numpy as np
import Filter_Code.find_passes as find_passes

game_ids = {'21500001', "21500002", "21500491"}

for game_id in game_ids:
    find_passes.update_json_with_actions(game_id)