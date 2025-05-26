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
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button  # <-- EKLENDİ

# Sabitler
class Constant:
    X_MIN = 0
    X_MAX = 94
    Y_MIN = 0
    Y_MAX = 50

# JSON verisini oku
with open("../data_result/match_logs/episode_6.json", "r") as f:
    data = json.load(f)

def get_coordinates(entry):
    state = entry[0]
    ball_x = state[3]
    ball_y = state[4]
    players = [(state[i], state[i+1]) for i in range(6, 6 + 20, 2)]  # 10 oyuncu
    possession = state[26]
    home_score = state[27]
    away_score = state[28]
    game_clock = state[1]
    shot_clock = state[2]
    return ball_x, ball_y, players, possession, home_score, away_score, game_clock, shot_clock

# Saha çizimi
fig, ax = plt.subplots(figsize=(Constant.X_MAX / 10, (Constant.Y_MAX + 10) / 10))  # Ekstra yukarı boşluk

# Court resmini ekle
court = plt.imread("court.png")
ax.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX, Constant.Y_MIN, Constant.Y_MAX])

# Çerçeve (saha kenarlığı)
border = Rectangle((Constant.X_MIN, Constant.Y_MIN), Constant.X_MAX, Constant.Y_MAX,
                   linewidth=2, edgecolor='black', facecolor='none', zorder=2)
ax.add_patch(border)

# Sınırlar ve ayarlar
ax.set_xlim(Constant.X_MIN, Constant.X_MAX)
ax.set_ylim(Constant.Y_MIN - 2, Constant.Y_MAX + 8)  # Ekstra üst boşluk
ax.set_aspect('equal')
ax.axis('off')

# Nesneler
ball, = ax.plot([], [], 'ro', markersize=5, zorder=3)
home_dots, = ax.plot([], [], 'bo', markersize=8, zorder=3)
away_dots, = ax.plot([], [], 'go', markersize=8, zorder=3)

# Üst merkeze sabit yazılar
score_text = ax.text((Constant.X_MIN + Constant.X_MAX) / 2, Constant.Y_MAX + 6,
                     '', fontsize=10, ha='center', weight='bold', zorder=4)
gameclock_text = ax.text((Constant.X_MIN + Constant.X_MAX) / 2, Constant.Y_MAX + 4,
                         '', fontsize=10, ha='center', weight='bold', zorder=4)
shotclock_text = ax.text((Constant.X_MIN + Constant.X_MAX) / 2, Constant.Y_MAX + 2,
                         '', fontsize=10, ha='center', weight='bold', zorder=4)

# SAHANIN DIŞINDA SAĞ ÜST KÖŞEYE ACTION TYPE YAZISI
action_text = ax.text(Constant.X_MAX + 2, Constant.Y_MAX + 6, '', 
                     fontsize=12, ha='right', va='top', color='purple', weight='bold', zorder=5)

current_frame = [0]  # Liste olarak tut, referansla değiştirilebilir

def update(frame):
    ball_x, ball_y, players, possession, home_score, away_score, game_clock, shot_clock = get_coordinates(data[frame])

    home_players = players[:5]
    away_players = players[5:]

    home_xs, home_ys = zip(*home_players)
    away_xs, away_ys = zip(*away_players)

    ball.set_data([ball_x], [ball_y])
    home_dots.set_data(home_xs, home_ys)
    away_dots.set_data(away_xs, away_ys)

    score_text.set_text(f'Possession: {possession} | Score: {home_score} - {away_score}')
    gameclock_text.set_text(f'Game Clock: {game_clock}')
    shotclock_text.set_text(f'Shot Clock: {shot_clock:.1f}')

    # ACTION TYPE YAZISINI GÜNCELLE
    action_type = data[frame][1] if len(data[frame]) > 1 else ""
    action_text.set_text(f'Action: {action_type}')

    fig.canvas.draw_idle()
    return ball, home_dots, away_dots, score_text, gameclock_text, shotclock_text, action_text

def next_frame(event):
    if current_frame[0] < len(data) - 1:
        current_frame[0] += 1
        update(current_frame[0])

def prev_frame(event):
    if current_frame[0] > 0:
        current_frame[0] -= 1
        update(current_frame[0])

# Butonlar için konumlar (saha dışında sağ alt köşe)
button_width = 0.08
button_height = 0.05
button_spacing = 0.01
right = 0.98
bottom = 0.02

axprev = plt.axes([right - 2*button_width - button_spacing, bottom, button_width, button_height])
axnext = plt.axes([right - button_width, bottom, button_width, button_height])

bnext = Button(axnext, 'İleri')
bprev = Button(axprev, 'Geri')

bnext.on_clicked(next_frame)
bprev.on_clicked(prev_frame)

# Başlangıç frame'i çiz
update(current_frame[0])

plt.show()
