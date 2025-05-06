# burasi bizim statelerimizi gorsellestiriyor. 
'''
[[period, game_time_remaining, game_clock, ball_x, ball_y, ball_z,
home1_x, home1_y, home2_x, home2_y, ..., home5_x, home5_y,
away1_x, away1_y, ..., away5_x, away5_y,
ball_possession, home_score, away_score], "action_type", action_result]
'''
 
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Örnek: JSON dosyasından veri yüklenmesi
with open("data/2016.NBA.Raw.SportVU.Game.Logs/21500491_last_result.json", "r") as f:
    data = json.load(f)

# Sahadaki öğe sayısı
NUM_PLAYERS = 10  # home5 + away5
BALL_INDEX = 5  # top x,y,z'den sonrakiler oyuncular

# Oyuncu koordinat indekslerini çıkar
def get_coordinates(entry):
    flat_data = entry[0]
    ball_x, ball_y = flat_data[3], flat_data[4]
    player_coords = []
    for i in range(BALL_INDEX + 1, BALL_INDEX + 1 + NUM_PLAYERS * 2, 2):
        player_coords.append((flat_data[i], flat_data[i+1]))
    return ball_x, ball_y, player_coords, flat_data[-3], flat_data[-2], flat_data[-1]  # top, oyuncular, possession, skorlar

# Grafik ayarları
fig, ax = plt.subplots()
court = plt.Rectangle((0, 0), 94, 50, linewidth=1, edgecolor='black', facecolor='none')
ax.add_patch(court)

ball, = ax.plot([], [], 'ko', label='Ball')
home_dots, = ax.plot([], [], 'ro', label='Home')
away_dots, = ax.plot([], [], 'bo', label='Away')
score_text = ax.text(2, 52, '', fontsize=12)

def init():
    ax.set_xlim(0, 94)
    ax.set_ylim(0, 50)
    ball.set_data([], [])
    home_dots.set_data([], [])
    away_dots.set_data([], [])
    score_text.set_text('')
    return ball, home_dots, away_dots, score_text

def update(frame):
    ball_x, ball_y, players, possession, home_score, away_score = get_coordinates(data[frame])
    home_players = players[:5]
    away_players = players[5:]

    home_xs, home_ys = zip(*home_players)
    away_xs, away_ys = zip(*away_players)

    ball.set_data([ball_x], [ball_y])  # <-- DÜZENLENDİ
    home_dots.set_data(home_xs, home_ys)
    away_dots.set_data(away_xs, away_ys)

    score_text.set_text(f'Possession: {possession} | Score: {home_score} - {away_score}')
    return ball, home_dots, away_dots, score_text


ani = animation.FuncAnimation(fig, update, frames=len(data), init_func=init, blit=True, interval=300)

plt.legend()
plt.title("Basketbol Hareketleri Animasyonu")
plt.show()
