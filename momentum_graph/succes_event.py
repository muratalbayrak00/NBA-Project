import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr

# JSON dosyasını okuma
with open("21500491_result.json", "r") as file:
    data = json.load(file)

# "Shot", "Pass" ve "Dribble" eylemleri olan oyunları seçme
game_with_actions = []

for game in data:
    i= 0
    moments = game["moments"]
    actions = [moment for moment in moments if moment.get("action") in ["shot", "pass", "dribble"]]
    if actions:
        game_with_actions.append({"game_id": i, "actions": actions})
    i = i+1
# Korelasyon hesaplamak için fonksiyon
def calculate_similarity(game1, game2):
    # Şut zamanları
    game1_shot_times = [moment['game_clock'] for moment in game1['actions'] if moment.get("action") == "shot"]
    game2_shot_times = [moment['game_clock'] for moment in game2['actions'] if moment.get("action") == "shot"]

    # Şut verisi yeterli mi?
    if len(game1_shot_times) < 2 or len(game2_shot_times) < 2:
        return float('inf')  # Yeterli şut verisi yoksa yüksek benzerlik değeri döndür

    # Korelasyonu hesapla
    correlation, _ = pearsonr(game1_shot_times, game2_shot_times)
    
    # Toplam benzerlik skoru (burada korelasyon değerini kullanabilirsiniz)
    return correlation

# En benzer oyunu bulma
min_similarity = float('inf')
most_similar_game = None

for i, game1 in enumerate(game_with_actions):
    for j, game2 in enumerate(game_with_actions):
        if i != j:
            similarity = calculate_similarity(game1, game2)
            if similarity < min_similarity:
                min_similarity = similarity
                most_similar_game = game2


# Basketbol sahası görseli (NBA ölçülerine uygun)
def plot_basketball_court():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Basketbol sahası çizimi (NBA ölçülerine uygun)
    ax.add_patch(Rectangle((0, 0), 28, 15, linewidth=2, edgecolor='black', facecolor='none'))  # Tam saha
    ax.add_patch(Rectangle((0, 7), 2, 1, linewidth=2, edgecolor='black', facecolor='none'))  # Ev sahibi potası
    ax.add_patch(Rectangle((28, 7), 2, 1, linewidth=2, edgecolor='black', facecolor='none'))  # Deplasman potası
    ax.add_patch(Rectangle((14 - 6, 0), 12, 15, linewidth=2, edgecolor='black', facecolor='none'))  # Orta saha çizgisi
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 15)
    
    return ax

def plot_actions_on_court(event, ax):
    shot_color = 'blue'
    pass_color = 'green'
    dribble_color = 'orange'
    
    # game objesinin yapısını kontrol et
    print(event.keys())

    # "moments" anahtarını alıyoruz
    moments = event.get("moments", [])
    
    # Eğer moments yoksa, işleme devam etme
    if not moments:
        print("No moments found in the game.")
        return
    
    for moment in moments:
        # "ball" anahtarının varlığını kontrol et
        if 'ball' in moment and len(moment['ball']) >= 4:
            x, y = moment['ball'][2], moment['ball'][3]
            
            # "action" anahtarının varlığını kontrol et
            if 'action' in moment:
                print(moment)
                action_type = moment['action']
                
                # Action türüne göre renk ve çizim
                if action_type == 'shot':
                    ax.plot(x, y, 'o', color=shot_color, markersize=8)
                elif action_type == 'pass':
                    ax.plot(x, y, 'o', color=pass_color, markersize=8)
                elif action_type == 'motion':
                    ax.plot(x, y, 'o', color=dribble_color, markersize=8)
                else:
                    print(f"Unknown action type: {action_type}")
            else:
                print("Action type missing in moment.")
        else:
            print("Ball coordinates missing in moment.")

# En benzer oyunun şutlarını basketbol sahasında gösterme
ax = plot_basketball_court()


# Ev sahibi takımının hareketlerini sahada gösterme
plot_actions_on_court(data[50], ax)

# Deplasman takımının hareketlerini sahada gösterme
plot_actions_on_court(game2, ax)

# Grafik başlığı ve etiketleri
ax.set_title("Most Similar Shot Game on Basketball Court", fontsize=16)
plt.show()
