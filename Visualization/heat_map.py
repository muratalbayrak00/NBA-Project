import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def draw_heat_maps_from_json(json_path, output_dir="heatmaps"):
    """
    json_path: str -> path to your game sequence data in JSON format
    output_dir: str -> where to save the heatmap figures
    """
    # Load JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract coordinates
    ball_positions = []
    shot_positions_home = []
    shot_positions_away = []
    home_team_positions = []
    away_team_positions = []

    for frame in data:
        frame_data, action_type, _ = frame

        ball_x = frame_data[3]
        ball_y = frame_data[4]
        ball_positions.append((ball_x, ball_y))

        # Shots
        if action_type == "shot":
            if frame_data[26] == 1:
                shot_positions_home.append((ball_x, ball_y))
            else:
                shot_positions_away.append((ball_x, ball_y))

        # Home team player positions (5 players)
        for i in range(5):
            home_team_positions.append((frame_data[6 + i * 2], frame_data[6 + i * 2 + 1]))

        # Away team player positions (5 players)
        for i in range(5):
            away_team_positions.append((frame_data[16 + i * 2], frame_data[16 + i * 2 + 1]))

    os.makedirs(output_dir, exist_ok=True)

    # A. Home Team Movement Heatmap
    if home_team_positions:
        x, y = zip(*home_team_positions)
        plt.figure(figsize=(8, 4))
        sns.kdeplot(x=x, y=y, cmap="Reds", fill=True, bw_adjust=0.3, thresh=0.05)
        plt.title("Home Team Movement Heatmap")
        plt.xlim(0, 94)
        plt.ylim(0, 50)
        plt.xlabel("Court Length")
        plt.ylabel("Court Width")
        plt.savefig(os.path.join(output_dir, "home_team_movement_heatmap.png"))
        plt.close()

    # B. Away Team Movement Heatmap
    if away_team_positions:
        x, y = zip(*away_team_positions)
        plt.figure(figsize=(8, 4))
        sns.kdeplot(x=x, y=y, cmap="Blues", fill=True, bw_adjust=0.3, thresh=0.05)
        plt.title("Away Team Movement Heatmap")
        plt.xlim(0, 94)
        plt.ylim(0, 50)
        plt.xlabel("Court Length")
        plt.ylabel("Court Width")
        plt.savefig(os.path.join(output_dir, "away_team_movement_heatmap.png"))
        plt.close()

    # C. Combined Movement Heatmap
    if home_team_positions and away_team_positions:
        x1, y1 = zip(*home_team_positions)
        x2, y2 = zip(*away_team_positions)
        plt.figure(figsize=(8, 4))
        sns.kdeplot(x=x1, y=y1, cmap="Reds", fill=True, bw_adjust=0.3, thresh=0.05, label="Home")
        sns.kdeplot(x=x2, y=y2, cmap="Blues", fill=True, bw_adjust=0.3, thresh=0.05, label="Away")
        plt.title("Combined Team Movement Heatmap")
        plt.xlim(0, 94)
        plt.ylim(0, 50)
        plt.xlabel("Court Length")
        plt.ylabel("Court Width")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "combined_movement_heatmap.png"))
        plt.close()

    # D. Shot Density Heatmap - Home
    if shot_positions_home:
        x, y = zip(*shot_positions_home)
        plt.figure(figsize=(8, 4))
        sns.kdeplot(x=x, y=y, cmap="Oranges", fill=True, bw_adjust=0.3, thresh=0.05)
        plt.title("Home Team Shot Density Heatmap")
        plt.xlim(0, 94)
        plt.ylim(0, 50)
        plt.xlabel("Court Length")
        plt.ylabel("Court Width")
        plt.savefig(os.path.join(output_dir, "home_shot_density_heatmap.png"))
        plt.close()

    # E. Shot Density Heatmap - Away
    if shot_positions_away:
        x, y = zip(*shot_positions_away)
        plt.figure(figsize=(8, 4))
        sns.kdeplot(x=x, y=y, cmap="Purples", fill=True, bw_adjust=0.3, thresh=0.05)
        plt.title("Away Team Shot Density Heatmap")
        plt.xlim(0, 94)
        plt.ylim(0, 50)
        plt.xlabel("Court Length")
        plt.ylabel("Court Width")
        plt.savefig(os.path.join(output_dir, "away_shot_density_heatmap.png"))
        plt.close()

    # F. Combined Shot Density Heatmap
    if shot_positions_home and shot_positions_away:
        x1, y1 = zip(*shot_positions_home)
        x2, y2 = zip(*shot_positions_away)
        plt.figure(figsize=(8, 4))
        sns.kdeplot(x=x1, y=y1, cmap="Oranges", fill=True, bw_adjust=0.3, thresh=0.05, label="Home")
        sns.kdeplot(x=x2, y=y2, cmap="Purples", fill=True, bw_adjust=0.3, thresh=0.05, label="Away")
        plt.title("Combined Shot Density Heatmap")
        plt.xlim(0, 94)
        plt.ylim(0, 50)
        plt.xlabel("Court Length")
        plt.ylabel("Court Width")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "combined_shot_density_heatmap.png"))
        plt.close()

    # G. Ball Movement Heatmap
    if ball_positions:
        x, y = zip(*ball_positions)
        plt.figure(figsize=(8, 4))
        sns.kdeplot(x=x, y=y, cmap="Greens", fill=True, bw_adjust=0.3, thresh=0.05)
        plt.title("Ball Movement Heatmap")
        plt.xlim(0, 94)
        plt.ylim(0, 50)
        plt.xlabel("Court Length")
        plt.ylabel("Court Width")
        plt.savefig(os.path.join(output_dir, "ball_movement_heatmap.png"))
        plt.close()

    print(f"Heatmaps saved to '{output_dir}'")

if __name__ == "__main__":
    file_path = '21500491_last_result.json'

    draw_heat_maps_from_json(file_path)
    