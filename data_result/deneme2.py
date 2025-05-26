import json
import os
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# ------------------ 1. Basketbol Ortamı ------------------ #
class BasketballEnv:
    # BasketballEnv sınıfına bu sabitleri ekleyin
    COURT_MIN_X = 0
    COURT_MAX_X = 94
    COURT_MIN_Y = 0
    COURT_MAX_Y = 50
    def __init__(self):
        self.state_dim = 29  # 29 elemanlı durum uzayı
        self.action_dim = 5  
        self.reset()
        self.pass_in_progress = False
        self.pass_distance_x = 0
        self.pass_distance_y = 0
        self.pass_steps_remaining = 0
        self.shot_in_progress = False
        self.shot_target = None
        self.shot_steps_remaining = 0
        self.shot_success = 0
        self.shot_points = 0
        self.shot_distance = 0 
        self.shot_distance_x = 0
        self.shot_distance_y = 0
        self.last_touch = 0
        self.force_pass = False
        self.force_shot = False
        self.away_shot = False
        self.away_pass = False
        self.home_shot = False
        self.home_pass = False
        self.ball_holder = 0 
        

    def reset(self):
        """Oyunu sıfırlar ve başlangıç durumunu döndürür."""
        self.state = [
            1.0,  # Periyot
            720.0,  # Kalan periyot süresi
            24.0,  # Kalan atak süresi
            47.0,  # Top X konumu (Sahanın ortası)
            25.0,  # Top Y konumu
            10.0   # Top Z konumu
        ]

        # Ev sahibi oyuncular (5 oyuncu, x: 0-47, y: 0-50)
        for _ in range(5):
            x = np.random.randint(0, 47)
            y = np.random.randint(0, 50)
            self.state.append(x)
            self.state.append(y)

        # Rakip oyuncular (5 oyuncu, x: 47-94, y: 0-50)
        for _ in range(5):
            x = np.random.randint(47, 94)
            y = np.random.randint(0, 50)
            self.state.append(x)
            self.state.append(y)

        # **Skor Verileri**
        self.state.append(0)  # Topa sahip takımın skoru
        self.state.append(0)    # Ev sahibi skor
        self.state.append(0)    # Deplasman skor

        return self.state
    
    def get_valid_actions(self, is_home_team):
        return [0, 1, 3]
    
    def select_pass_target(self, state, ball_x, ball_y):
        start_idx = 6
        end_idx = 16
        defender_start = 16
        defender_end = 26

        # Top sahibi oyuncunun indexini bul
        owner_idx = None
        min_dist = float('inf')
        for i in range(start_idx, end_idx, 2):
            px, py = state[i], state[i+1]
            dist = math.sqrt((px - ball_x)**2 + (py - ball_y)**2)
            if dist < min_dist:
                min_dist = dist
                owner_idx = i

        candidates = []
        # Adayları belirle: topun ilerisinde olanlar ve top sahibi hariç
        for i in range(start_idx, end_idx, 2):
            if i == owner_idx:
                continue
            x = state[i]
            y = state[i+1]
            if y > ball_y:  # sadece topun ilerisindekileri al
                # Bu oyuncuya en yakın savunmacının mesafesini bul
                min_def_dist = float('inf')
                for j in range(defender_start, defender_end, 2):
                    def_x, def_y = state[j], state[j+1]
                    def_dist = math.sqrt((x - def_x)**2 + (y - def_y)**2)
                    if def_dist < min_def_dist:
                        min_def_dist = def_dist
                # Hem topa uzaklığı hem savunmacıya uzaklığı ile aday listesine ekle
                candidates.append((min_def_dist, -math.sqrt((x - ball_x)**2 + (y - ball_y)**2), i))

        # Eğer topun ilerisinde kimse yoksa, tüm takım arkadaşlarını aday olarak al
        if not candidates:
            for i in range(start_idx, end_idx, 2):
                if i == owner_idx:
                    continue
                x = state[i]
                y = state[i+1]
                min_def_dist = float('inf')
                for j in range(defender_start, defender_end, 2):
                    def_x, def_y = state[j], state[j+1]
                    def_dist = math.sqrt((x - def_x)**2 + (y - def_y)**2)
                    if def_dist < min_def_dist:
                        min_def_dist = def_dist
                candidates.append((min_def_dist, -math.sqrt((x - ball_x)**2 + (y - ball_y)**2), i))

        # En uygun adayı seç: önce savunmacıya uzaklık, eşitlik varsa topa yakınlık
        candidates.sort(reverse=True)
        return candidates[0][2]  # index'i döndür
        
    
    def assign_ball_to_nearest_player(self, state):
        """
        Topa 3 feet (0.91 metre) veya daha yakın olan oyuncular arasında en yakına topu verir ve state[26]'yı günceller.
        """
        ball_x, ball_y = state[3], state[4]
        nearest_dist = float('inf')
        nearest_team = None
        nearest_idx = None

        # Ev sahibi oyuncular (6-16), rakip oyuncular (16-26)
        for idx, team in [(range(6, 16, 2), 1), (range(16, 26, 2), 2)]:
            for i in idx:
                player_x, player_y = state[i], state[i+1]
                distance = math.sqrt((player_x - ball_x)**2 + (player_y - ball_y)**2)
                if distance <= 3 and distance < nearest_dist:
                    nearest_dist = distance
                    nearest_team = team
                    nearest_idx = i

        if nearest_team is not None and nearest_idx is not None:
            state[26] = nearest_team
            # Topu o oyuncunun üstüne koymak istersen:
            state[3] = state[nearest_idx] + 1
            state[4] = state[nearest_idx + 1] + 1
            state[5] = 5.0  # Top yerde
        else:
            state[26] = 0

        return state
    
    def set_away_zone_defense(self, state):
        """
        Away takım oyuncuları (index 5-9) klasik 2-3 alan savunması bölgelerine yerleşir.
        """
        # Alan savunması bölgelerinin merkez noktaları (örnek: x, y)
        zone_centers = [
            (60, 41),  # Guard 1 (sol üst)
            (81, 41),  # Guard 2 (sağ üst)
            (60, 10),  # Forward 1 (sol alt)
            (81, 10),  # Forward 2 (sağ alt)
            (70, 25),  # Center (pota önü)
        ]
        ball_x = state[3]
        ball_y = state[4]
        min_distance = float('inf')
        closest_defender_idx = None
        # En yakın savunmacıyı bul
        for i in range(16, 26, 2):
            dx = ball_x - state[i]
            dy = ball_y - state[i+1]
            distance = math.sqrt(dx**2 + dy**2)
            if distance < min_distance:
                min_distance = distance
                closest_defender_idx = i

        # En yakın savunmacı topa doğru koşsun
        if closest_defender_idx is not None:
            dx = ball_x - state[closest_defender_idx]
            dy = ball_y - state[closest_defender_idx + 1]
            norm = math.sqrt(dx**2 + dy**2)
            if norm != 0:
                dx /= norm
                dy /= norm
                state[closest_defender_idx] += dx * np.random.uniform(2.0, 3.0) # hızını değiştirebilirsin
                state[closest_defender_idx + 1] += dy * np.random.uniform(2.0, 3.0)

        # Away oyuncularının indexleri: 5 oyuncu, state[16], state[18], ..., state[24]
        for idx, (zone_x, zone_y) in zip(range(16, 26, 2), zone_centers):
            if idx == closest_defender_idx:
                continue
            px, py = state[idx], state[idx+1]
            dx = zone_x - px
            dy = zone_y - py
            norm = (dx**2 + dy**2) ** 0.5
            if norm > 0:
                dx /= norm
                dy /= norm
                # Alan merkezine doğru hareket (hız isteğe göre ayarlanabilir)
                state[idx] += dx * np.random.uniform(0.5, 2.5)
                state[idx+1] += dy * np.random.uniform(0.5, 2.5)
        return state
    
    def set_home_zone(self, state):
        zone_centers = [
            (60, 41),  # Guard 1 (sol üst)
            (81, 41),  # Guard 2 (sağ üst)
            (60, 10),  # Forward 1 (sol alt)
            (81, 10),  # Forward 2 (sağ alt)
            (70, 25),  # Center (pota önü)
        ]
         # Top sahibi oyuncunun indexini bul
        ball_x, ball_y = state[3], state[4]
        owner_idx = None
        min_dist = float('inf')
        for i in range(6, 16, 2):
            px, py = state[i], state[i+1]
            dist = math.sqrt((px - ball_x)**2 + (py - ball_y)**2)
            if dist < min_dist:
                min_dist = dist
                owner_idx = i
        
        for idx, (zone_x, zone_y) in zip(range(6, 16, 2), zone_centers):
            if i == owner_idx:
                continue
            px, py = state[idx], state[idx+1]
            dx = zone_x - px
            dy = zone_y - py
            norm = (dx**2 + dy**2) ** 0.5
            if norm > 0:
                dx /= norm
                dy /= norm
                # Alan merkezine doğru hareket (hız isteğe göre ayarlanabilir)
                state[idx] += dx * np.random.uniform(0.5, 2.5)
                state[idx+1] += dy * np.random.uniform(0.5, 2.5)
        return state
    
    def closest_defender_runs_to_ball(self, state):
        """
        Rakip takım oyuncularından topa en yakın olanı topa doğru koşar.
        """
        ball_x = state[3]
        ball_y = state[4]
        defender_start = 16
        defender_end = 26
        attacker_start = 6
        attacker_end = 16

        min_distance = float('inf')
        closest_defender_idx = None

        # En yakın savunmacıyı bul
        for i in range(defender_start, defender_end, 2):
            dx = ball_x - state[i]
            dy = ball_y - state[i+1]
            distance = math.sqrt(dx**2 + dy**2)
            if distance < min_distance:
                min_distance = distance
                closest_defender_idx = i

        # En yakın savunmacı topa doğru koşsun
        if closest_defender_idx is not None:
            dx = ball_x - state[closest_defender_idx]
            dy = ball_y - state[closest_defender_idx + 1]
            norm = math.sqrt(dx**2 + dy**2)
            if norm != 0:
                dx /= norm
                dy /= norm
                state[closest_defender_idx] += dx * np.random.uniform(2.0, 3.0) # hızını değiştirebilirsin
                state[closest_defender_idx + 1] += dy * np.random.uniform(2.0, 3.0)

        #Diğer savunmacılar kendilerine en yakın hücum oyuncusunu tutsun
        for i in range(defender_start, defender_end, 2):
            if i == closest_defender_idx:
                continue
            # En yakın hücum oyuncusunu bul
            min_att_dist = float('inf')
            closest_att_idx = None
            for j in range(attacker_start, attacker_end, 2):
                att_dx = state[j] - state[i]
                att_dy = state[j+1] - state[i+1]
                att_dist = math.sqrt(att_dx**2 + att_dy**2)
                if att_dist < min_att_dist:
                    min_att_dist = att_dist
                    closest_att_idx = j
            # Ona doğru yaklaş
            if closest_att_idx is not None:
                dx = state[closest_att_idx] - state[i]
                dy = state[closest_att_idx + 1] - state[i+1]
                norm = math.sqrt(dx**2 + dy**2)
                if norm != 0:
                    dx /= norm
                    dy /= norm
                    state[i] += dx * np.random.uniform(1, 2.5)
                    state[i+1] += dy * np.random.uniform(1, 2.5)
        return state
    
    def home_players_cut_to_open_space(self, state):
        """
        Ev sahibi takım oyuncuları (top sahibi hariç) x ekseninde 47-94 arasındaki en boş alanlara hareket eder.
        """
        start_idx = 6
        end_idx = 16

        # Top sahibi oyuncunun indexini bul
        ball_x, ball_y = state[3], state[4]
        owner_idx = None
        min_dist = float('inf')
        for i in range(start_idx, end_idx, 2):
            px, py = state[i], state[i+1]
            dist = math.sqrt((px - ball_x)**2 + (py - ball_y)**2)
            if dist < min_dist:
                min_dist = dist
                owner_idx = i

        # Grid oluştur (x: 47-94 arası, y: tüm saha)
        grid_x = np.linspace(47, self.COURT_MAX_X - 2, 10)
        grid_y = np.linspace(self.COURT_MIN_Y + 2, self.COURT_MAX_Y - 2, 5)
        grid_points = [(x, y) for x in grid_x for y in grid_y]

        # Diğer oyuncuların pozisyonlarını topla
        all_players = []
        for i in range(6, 26, 2):
            all_players.append((state[i], state[i+1]))

        # Her oyuncu için en boş noktayı bul ve oraya yönlendir
        for i in range(start_idx, end_idx, 2):
            if i == owner_idx:
                continue
            # Her grid noktası için diğer oyunculara toplam uzaklığı hesapla
            best_point = None
            best_score = -float('inf')
            for gx, gy in grid_points:
                total_dist = 0
                for px, py in all_players:
                    total_dist += math.sqrt((gx - px)**2 + (gy - py)**2)
                if total_dist > best_score:
                    best_score = total_dist
                    best_point = (gx, gy)
            # En boş noktaya doğru hareket et
            if best_point is not None:
                dx = best_point[0] - state[i]
                dy = best_point[1] - state[i+1]
                norm = math.sqrt(dx**2 + dy**2)
                if norm != 0:
                    dx /= norm
                    dy /= norm
                    state[i] += dx * np.random.uniform(0.5, 2.5)
                    state[i+1] += dy * np.random.uniform(0.5, 2.5)
        return state
    
    def find_closest_defender_to_ball(self, state):
        """
        Topa en yakın rakip (defans) oyuncunun indexini ve mesafesini döndürür.
        """
        ball_x, ball_y = state[3], state[4]
        defender_start = 16
        defender_end = 26

        min_dist = float('inf')
        closest_idx = None
        for i in range(defender_start, defender_end, 2):
            dx = state[i] - ball_x
            dy = state[i+1] - ball_y
            dist = math.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        return closest_idx
    
    def step(self, action):
        
        new_state = self.state.clone()
        reward = 0
        # **Süre azalımı**
        new_state[1] -= 0.2  # Periyot süresi azalır
        new_state[2] -= 0.2  # Atak süresi azalır
        if action != 4:
            new_state = self.set_home_zone(new_state)
            #new_state = self.closest_defender_runs_to_ball(new_state) 
            new_state = self.set_away_zone_defense(new_state) 
        

        ball_x = new_state[3]
        ball_y = new_state[4]
        ball_owner = new_state[26]
        if ball_owner == 1:
            self.last_touch = 1
        elif ball_owner == 2:
            self.last_touch = 2
           
        if new_state[2] <=0:
            new_owner_start = 16
            if ball_owner == 0:
                if self.last_touch == 2:
                    new_owner_start = 6
            margin = 0.5  # Kenardan içeride başlasın
            x, y = new_state[3], new_state[4]

            # Kenarlara uzaklıkları hesapla
            dist_left = abs(x - self.COURT_MIN_X)
            dist_right = abs(x - self.COURT_MAX_X)
            dist_top = abs(y - self.COURT_MIN_Y)
            dist_bottom = abs(y - self.COURT_MAX_Y)

            min_dist = min(dist_left, dist_right, dist_top, dist_bottom)

            # Topu en yakın kenara taşı
            if min_dist == dist_left:
                x_new = self.COURT_MIN_X + margin
                y_new = y
            elif min_dist == dist_right:
                x_new = self.COURT_MAX_X - margin
                y_new = y
            elif min_dist == dist_top:
                x_new = x
                y_new = self.COURT_MIN_Y + margin
            else:  # min_dist == dist_bottom
                x_new = x
                y_new = self.COURT_MAX_Y - margin

            new_state[3] = x_new
            new_state[4] = y_new
            new_state[5] = 5.0
            new_state[new_owner_start] = new_state[3]
            new_state[new_owner_start + 1] = new_state[4]
            self.force_pass = True
            # Atak süresini sıfırla
            new_state[2] = 24.0
            reward = -1
            #reward += -0.1  # Atak süresi dolduğunda ceza
            done = new_state[1] <= 0
            return new_state, reward, done

        if((new_state[3] < self.COURT_MIN_X) or (new_state[3] > self.COURT_MAX_X) or (new_state[4] < self.COURT_MIN_Y) or (new_state[4] > self.COURT_MAX_Y)):
            new_owner_start = 16
            if ball_owner == 0:
                if self.last_touch == 2:
                    new_owner_start = 6
            x, y = new_state[3], new_state[4]
            if x<0 and y<0:
                x_new = 1
                y_new = 1
            elif x<0 and y>50:
                x_new = 1
                y_new = 49
            elif x>94 and y<0:
                x_new = 93
                y_new = 1   
            elif x>94 and y>50:     
                x_new = 93
                y_new = 49
            elif x<0 and (y>=0 and y<=50):
                x_new = 1
                y_new = y
            elif x>94 and (y>=0 and y<=50): 
                x_new = 93
                y_new = y
            elif y<0 and (x>=0 and x<=94):
                x_new = x
                y_new = 1
            elif y>50 and (x>=0 and x<=94):
                x_new = x
                y_new = 49

            new_state = self.set_home_zone(new_state)
            #new_state = self.closest_defender_runs_to_ball(new_state)
            new_state = self.set_away_zone_defense(new_state)
            new_state[3] = x_new
            new_state[4] = y_new
            new_state[5] = 5.0
            new_state[new_owner_start] = new_state[3]
            new_state[new_owner_start + 1] = new_state[4]
            new_state[2] = 24.0
            self.force_pass = True
            reward = -1
            #reward += -0.1 
            done = new_state[1] <= 0 
            return new_state, reward, done
        
        if self.ball_holder != self.last_touch:
            new_state[2] = 24.0  # Atak süresi sıfırlanır
            self.ball_holder = self.last_touch
            reward = -1  # Top sahibi değiştiğinde ödül

        if self.pass_in_progress:
            # Topu hedefe doğru hareket ettir
            if self.pass_steps_remaining > 1:
                new_state[3] += self.pass_distance_x
                new_state[4] += self.pass_distance_y
                self.force_pass = True
                self.pass_steps_remaining -= 1
            else:
                new_state[3] += self.pass_distance_x + 1
                new_state[4] += self.pass_distance_y + 1
                self.pass_in_progress = False
                self.home_pass = False
                self.away_pass = False
                self.pass_target = None
                #reward += 0.1
            #new_state = self.home_players_cut_to_open_space(new_state)
            #new_state = self.closest_defender_runs_to_ball(new_state)
            done = new_state[1] <= 0
            return new_state, reward, done

        if self.shot_in_progress:
            # Topu potaya doğru hareket ettir
            if self.shot_steps_remaining > 0:
                new_state[3] += self.shot_distance_x
                new_state[4] += self.shot_distance_y
                self.force_shot = True
                self.shot_steps_remaining -= 1
            else:
                
                if self.shot_success:
                    #  Skor güncellenir
                    if self.home_shot:
                        new_state[27] += self.shot_points  # Ev sahibi
                    elif self.away_shot:
                        new_state[28] += self.shot_points  # Deplasman
                    #reward += self.shot_points * 3

                    #new_state = self.home_players_cut_to_open_space(new_state)
                    #new_state = self.closest_defender_runs_to_ball(new_state)
                    new_state[16] = 93
                    new_state[17] = 27
                    new_state[3] = 93
                    new_state[4] = 26
                    new_state[5] = 5.0
                    new_state[2] = 24.0
                    reward = 1
                    self.force_pass = True
                else:
                    if self.shot_distance > 47:
                        new_state[3] = 96 
                        new_state[4] = 25 
                        new_state[5] = 5.0
                    else:
                        #  Şut başarısız: top sekiyor (rastgele pozisyona düşüyor)
                        sekme_çarpanı = min(self.shot_distance / 10, 1.5)
                        new_state[3] = 88 + np.random.uniform(-5, 5) * sekme_çarpanı
                        new_state[4] = 25 + np.random.uniform(-5, 5) * sekme_çarpanı
                        new_state[5] = 5.0
                    #new_state = self.home_players_cut_to_open_space(new_state)
                    #new_state = self.closest_defender_runs_to_ball(new_state)
                    #reward += -0.1  # Başarısız şut için ceza
                self.shot_in_progress = False
                self.away_shot = False
                self.home_shot = False
            done = new_state[1] <= 0
            return new_state, reward, done

        if action == 0:  # PASS
            #pass_success = random.random() < 0.83
            pass_success = True
            if pass_success:
                target_index = self.select_pass_target(new_state, ball_x, ball_y)
            else:
                target_index = self.find_closest_defender_to_ball(new_state)
                
            
            target_x = new_state[target_index] 
            target_y = new_state[target_index + 1]
            self.pass_distance_x = (target_x - new_state[3]) * 0.33
            self.pass_distance_y = (target_y - new_state[4]) * 0.33
            self.pass_in_progress = True
            self.pass_steps_remaining = 2
            self.force_pass = True
            new_state[3] += self.pass_distance_x
            new_state[4] += self.pass_distance_y
            #new_state = self.closest_defender_runs_to_ball(new_state)
            #new_state = self.set_away_zone_defense(new_state)
            #new_state = self.home_players_cut_to_open_space(new_state)
            #new_state = self.set_home_zone_defense(new_state)
            
        elif action == 1:  # SHOT
            hoop_x = 88
            hoop_y = 25
            distance = math.sqrt((new_state[3] - hoop_x)**2 + (new_state[4] - hoop_y)**2)
            defender_start = 16 
            defender_end = 26  
            min_defender_distance = float('inf')

            for i in range(defender_start, defender_end, 2):
                dx = new_state[3] - new_state[i]
                dy = new_state[4] - new_state[i+1]
                d = math.sqrt(dx**2 + dy**2)
                if d < min_defender_distance:
                    min_defender_distance = d

            # Başarı olasılığı hesapla
            if distance < 20:
                base_prob = 0.5
                points = 2
            elif distance < 25:
                base_prob = 0.35
                points = 3
            else:
                base_prob = 0.001
                points = 3 if distance >= 25 else 2

            # Savunma etkisi: Yakın savunma varsa başarı oranı düşer
            if min_defender_distance < 2:
                base_prob -= 0.15
            elif min_defender_distance < 5:
                base_prob -= 0.05

            self.shot_distance_x = (hoop_x - new_state[3]) * 0.33
            self.shot_distance_y = (hoop_y - new_state[4]) * 0.33
            self.shot_success = random.random() < base_prob
            self.shot_points = points
            self.shot_in_progress = True
            self.shot_steps_remaining = 2
            self.force_shot = True
            self.shot_distance = distance
            new_state[3] += self.shot_distance_x
            new_state[4] += self.shot_distance_y
            #new_state = self.closest_defender_runs_to_ball(new_state)
            #new_state = self.set_away_zone_defense(new_state)
            #new_state = self.home_players_cut_to_open_space(new_state)
            #new_state = self.set_home_zone_defense(new_state)

        elif action == 3:  # dribble
            # Topun pozisyonu
            ball_x = new_state[3]
            ball_y = new_state[4]

            # Potanın koordinatları
            hoop_x = 88
            hoop_y = 25

            # Topu süren oyuncunun indexini bul (kendi takımından ve topa en yakın)
            player_range = range(6, 16, 2)
            min_dist = float('inf')
            owner_idx = None
            for i in player_range:
                px, py = new_state[i], new_state[i+1]
                dist = math.sqrt((px - ball_x)**2 + (py - ball_y)**2)
                if dist < min_dist:
                    min_dist = dist
                    owner_idx = i

            # Potaya doğru yön
            direction_x = hoop_x - ball_x
            direction_y = hoop_y - ball_y
            norm = math.sqrt(direction_x**2 + direction_y**2)
            if norm != 0:
                direction_x /= norm
                direction_y /= norm

            move_x = direction_x * 3
            move_y = direction_y * 3
            #new_state = self.closest_defender_runs_to_ball(new_state)
            #new_state = self.set_away_zone_defense(new_state)
            #new_state = self.home_players_cut_to_open_space(new_state)
            #new_state = self.set_home_zone_defense(new_state)
            # Top ve topu süren oyuncu birlikte hareket ediyor
            new_state[3] += move_x
            new_state[4] += move_y
            new_state[5] = 5.0  # top yerde

            if owner_idx is not None:
                new_state[owner_idx] = new_state[3] + 0.2
                new_state[owner_idx + 1] = new_state[4] + 0.2

            #reward += 0.01

        elif action == 4:
            zone_centers_1 = [
            (60, 41),  # Guard 1 (sol üst)
            (81, 41),  # Guard 2 (sağ üst)
            (60, 25),  # Forward 1 (sol alt)
            (81, 25),  # Forward 2 (sağ alt)
            (70, 10),  # Center (pota önü)
            ]
            zone_centers_2 = [
            (34, 41),  # Guard 1 (sol üst)
            (13, 41),  # Guard 2 (sağ üst)
            (34, 25),  # Forward 1 (sol alt)
            (13, 25),  # Forward 2 (sağ alt)
            (24, 10),  # Center (pota önü)
            ]
            
            ball_x = new_state[3]
            ball_y = new_state[4]
            if ball_x < 47:
                target_points = zone_centers_2
            else:
                target_points = zone_centers_1
            # En yakın ev sahibi oyuncu
            min_dist_home = float('inf')
            closest_home = None
            for i in range(6, 16, 2):
                dx = ball_x - new_state[i]
                dy = ball_y - new_state[i+1]
                dist = math.sqrt(dx**2 + dy**2)
                if dist < min_dist_home:
                    min_dist_home = dist
                    closest_home = i
            # En yakın deplasman oyuncusu
            min_dist_away = float('inf')
            closest_away = None
            for i in range(16, 26, 2):
                dx = ball_x - new_state[i]
                dy = ball_y - new_state[i+1]
                dist = math.sqrt(dx**2 + dy**2)
                if dist < min_dist_away:
                    min_dist_away = dist
                    closest_away = i
            # En yakınlar topa koşsun
            for idx in [closest_home, closest_away]:
                if idx is not None:
                    dx = ball_x - new_state[idx]
                    dy = ball_y - new_state[idx+1]
                    norm = math.sqrt(dx**2 + dy**2)
                    if norm != 0:
                        dx /= norm
                        dy /= norm
                        new_state[idx] += dx * np.random.uniform(1.5, 2.5)
                        new_state[idx+1] += dy * np.random.uniform(1.5, 2.5)

            for i in range(6, 26, 2):
                if i == closest_home or i == closest_away:
                    continue
                # Sahada rastgele bir hedef belirle (örneğin sahanın ortası veya kenarları)
                target_x, target_y = target_points[idx % len(target_points)]
                dx = target_x - new_state[i]
                dy = target_y - new_state[i+1]
                norm = math.sqrt(dx**2 + dy**2)
                if norm != 0:
                    dx /= norm
                    dy /= norm
                    new_state[i] += dx * np.random.uniform(0.5, 2.5)
                    new_state[i+1] += dy * np.random.uniform(0.5, 2.5)

            # Topun pozisyonu biraz rastgele değişsin (sekiyor gibi)
            new_state[3] += np.random.uniform(-0.5, 0.5)
            new_state[4] += np.random.uniform(-0.5, 0.5)
            new_state[5] += np.random.uniform(0, 0.5)
        else:
            print("hatalı action")
            print (action)

        #new_state = self.assign_ball_to_nearest_player(new_state)
        done = new_state[1] <= 0 
        return new_state, reward, done   
 
# ------------------ 2. DQN Ajan Modeli ------------------ #
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ------------------ 3. Offline Verinin Yüklenmesi ------------------ #
def load_offline_data(file_path):
    """
    Offline verinizin her örneği [state, action, reward] formatında olmalıdır.
    Örneğin:
    [
      [[...29 elemanlı state...], "shot", 2],
      [[...29 elemanlı state...], "pass", 10],
      ...
    ]
    """
    action_map = {
        "pass": 0,
        "shot": 1,
        "defend": 2,
        "dribble": 3,
        "": 4  # boş aksiyon
    }

    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Eylemleri sayısal değerlere dönüştür
    for sample in data:
        state, action, reward = sample
        action = action_map.get(action, 4)  # Eğer tanımlanmış değilse, varsayılan olarak 4 (boş aksiyon) kabul edilir
        sample[1] = action  # Aksiyonu sayısal değere dönüştürüp güncelle
    
    return data

# ------------------ 4. Eğitim Parametreleri ------------------ #
env = BasketballEnv()
state_dim = env.state_dim
action_dim = env.action_dim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn = DQN(state_dim, action_dim).to(device)
optimizer = optim.Adam(dqn.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
memory = deque(maxlen=10000)

gamma = 0.95  
epsilon_home = 0.15   # Ev sahibi için düşük
epsilon_away = 0.35
epsilon_min = 0.01 
epsilon_decay = 0.995  
batch_size = 32  

def mirror_period_states_if_needed(period_data):
    """
    Eğer periyot başındaki oyuncuların ortalaması sağ yarı sahadaysa,
    periyottaki tüm state'lerin top ve oyuncu x konumlarını simetrik yapar.
    """
    if len(period_data) == 0:
        return period_data
    first_state = period_data[0][0]
    player_xs = [first_state[i] for i in range(6, 16, 2)]
    avg_x = sum(player_xs) / len(player_xs)
    if avg_x > 47:
        for sample in period_data:
            state = sample[0]
            # Topun x'i
            state[3] = 94 - state[3]
            # Oyuncuların x'leri
            for i in range(6, 26, 2):
                state[i] = 94 - state[i]
    return period_data

def split_by_period(data):
    """
    Veriyi periyotlara böler. Her periyot ayrı bir liste olarak döner.
    """
    periods = []
    current = []
    last_period = None
    for sample in data:
        period = sample[0][0]
        if last_period is not None and period != last_period:
            periods.append(current)
            current = []
        current.append(sample)
        last_period = period
    if current:
        periods.append(current)
    return periods
""""
# ------------------ 5. Offline Pretraining ------------------ #
offline_data_dir = "Last_result_data"
offline_files = [f for f in os.listdir(offline_data_dir) if f.endswith(".json")]

total_offline_samples = 0
for file in offline_files:
    offline_data_path = os.path.join(offline_data_dir, file)
    try:
        offline_data = load_offline_data(offline_data_path)
        periods = split_by_period(offline_data)
        for period_data in periods:
            # Periyot başında simetri gerekiyorsa tüm periyot state'lerine uygula
            period_data = mirror_period_states_if_needed(period_data)
            for sample in period_data:
                state, action, reward = sample
                state_tensor = torch.FloatTensor(state).to(device)
                next_state_tensor = torch.FloatTensor(state).to(device)
                done = True
                memory.append((state_tensor, action, reward, next_state_tensor, done))
            total_offline_samples += len(period_data)
        #print(f"{file} dosyasından {len(offline_data)} deney eklendi.")
    except Exception as e:
        print(f"{file} yüklenemedi:", e)

print(f"Toplam {total_offline_samples} offline deney replay buffer'a eklendi.")

# Offline pretraining için belirli adım sayısı (örneğin 1000 iterasyon)
pretrain_steps = 1
for step in range(pretrain_steps):
    if len(memory) < batch_size:
        break  # Yeterli veri yoksa pretrain yapılamaz
    
    minibatch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states = torch.stack(states)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.stack(next_states)
    dones = torch.BoolTensor(dones).to(device)

    q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    target_q_values = rewards

    loss = loss_fn(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (step + 1) % 200 == 0:
        print(f"Offline pretraining step: {step + 1}/{pretrain_steps}")

"""
# ------------------ 6. Online Eğitim ------------------ #
log_dir = "match_logs"
os.makedirs(log_dir, exist_ok=True)
def round_state(state):
    return [round(float(x), 2) for x in state]
def find_action(action):
    action_map = {
        0: "pass",
        1: "shot",
        2: "defend",
        3: "dribble",
        4: ""
    }
    return action_map.get(action, "unknown")
episodes = 1000  # Online eğitim için epizod sayısı
for episode in range(episodes):
    state = env.reset()
    env.state = torch.FloatTensor(state).to(device)  # state'i doğru cihaza taşı
    total_reward = 0
    done = False
    match_states = []
    while not done:
        valid_actions = env.get_valid_actions(is_home_team=True)
        ball_control = env.state[26]
        if env.away_shot or env.away_pass:
            if ball_control == 1:
                env.away_shot = False
                env.away_pass = False
                env.pass_in_progress = False
                env.shot_in_progress = False
                env.force_pass = False
                env.force_shot = False
            else:
                ball_control = 2
            #print(ball_control)
        if env.home_shot or env.home_pass:
            if ball_control == 2:
                env.home_shot = False
                env.home_pass = False
                env.pass_in_progress = False
                env.shot_in_progress = False
                env.force_pass = False
                env.force_shot = False
            else:
                ball_control = 1
            #print(ball_control)

        if ball_control == 1: 
            #print("Top Hoem da") # Ev sahibi
            #print(env.state)
            if env.force_pass:
                action = 0
                env.force_pass = False
            elif env.force_shot:
                action = 1
                env.force_shot = False
            else:
                if random.random() < epsilon_home:
                    action = random.choice(valid_actions)
                else:
                    with torch.no_grad():
                        q_values = dqn(env.state)
                        q_values_valid = q_values[valid_actions]
                        max_idx = torch.argmax(q_values_valid).item()
                        action = valid_actions[max_idx]
            if action == 1:
                env.home_shot = True
            if action == 0:
                env.home_pass = True          
            next_state, reward, done = env.step(action)  # Bu kısımda yeni state alınır
            
        elif ball_control == 2:  # Rakip
            temp = env.state[6:16].clone()
            env.state[6:16] = env.state[16:26]
            env.state[16:26] = temp
            #print(env.state)
            env.state[3] = 47 * 2 - env.state[3]
            for i in range(6, 26, 2):
                env.state[i] = 47 * 2 - env.state[i]  # x pozisyonunu simetrik yap

            if env.force_pass:
                action = 0
                env.force_pass = False
            elif env.force_shot:
                action = 1
                env.force_shot = False
            else:
                if random.random() < epsilon_away:
                    action = random.choice(valid_actions)
                else:
                    with torch.no_grad():
                        q_values = dqn(env.state)
                        q_values_valid = q_values[valid_actions]
                        max_idx = torch.argmax(q_values_valid).item()
                        action = valid_actions[max_idx]
            if action == 1:
                env.away_shot = True
            if action == 0:
                env.away_pass = True
        
            next_state, reward, done = env.step(action)  # Bu kısımda yeni state alınır
            temp = next_state[6:16].clone()
            next_state[6:16] = next_state[16:26]
            next_state[16:26] = temp
            next_state[3] = 47 * 2 - next_state[3]
            for i in range(6, 26, 2):
                next_state[i] = 47 * 2 - next_state[i]  # x pozisyonunu simetrik yap
            #print(next_state)    
            
        elif ball_control == 0:
            #print("Top dışarıda")
            action = 4
            next_state, reward, done = env.step(action)
        
        next_state = env.assign_ball_to_nearest_player(next_state)
        next_state = next_state.to(device)
        memory.append((env.state, action, reward, next_state, done))  # Memory'e ekle
        env.state = next_state  # State güncellenir (oyun bir sonraki adıma geçer)
        total_reward += reward
        state_list = env.state.cpu().numpy().tolist()
        match_states.append([round_state(state_list), [find_action(action)], [reward]])


        # Online öğrenme adımı
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)

            states = torch.stack(states).to(device)  # Cihaza taşımayı unutma
            actions = torch.LongTensor(actions).to(device)
            rewards = torch.FloatTensor(rewards).to(device)
            next_states = torch.stack(next_states).to(device)  # Cihaza taşımayı unutma
            dones = torch.BoolTensor(dones).to(device)

            q_values = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_values = dqn(next_states).max(1)[0].detach()
            target_q_values = rewards + (gamma * next_q_values * ~dones)

            loss = loss_fn(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    
    # Skor bilgisini al (state'in son iki elemanı)
    score = env.state.cpu().numpy()[-2:]  # CPU'ya taşı ve numpy array'e çevir
    home_score, away_score = score[0], score[1]

    # Epsilon değerini azalt
    epsilon_home = max(epsilon_min, epsilon_home * epsilon_decay)
    epsilon_away = max(epsilon_min, epsilon_away * epsilon_decay)
    if (episode + 1) % 10 == 1:
        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward:.2f}, Epsilon Home: {epsilon_home:.4f},Epsilon Away: {epsilon_away:.4f}, Score: {home_score:.0f}-{away_score:.0f}")
        with open(os.path.join(log_dir, f"dqn_1000_episode_{episode+1}.json"), "w") as f:json.dump(match_states, f)
# Eğitim tamamlandıktan sonra modeli kaydet
torch.save(dqn.state_dict(), "dqn_model_1000.pth")
print("Model başarıyla kaydedildi: dqn_model_1000.pth")



# ------------------ 7. Eğitim Sonrası Test ------------------ #
def evaluate_action_accuracy(model, data, device):
    """
    Modelin aksiyon doğruluğunu (accuracy) hesaplar.
    data: [ [state, gerçek_aksiyon, ...], ... ]
    """
    correct = 0
    total = 0
    for sample in data:
        state = torch.FloatTensor(sample[0]).to(device)
        true_action = sample[1]
        with torch.no_grad():
            q_values = model(state)
            predicted_action = torch.argmax(q_values).item()
        if predicted_action == true_action:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0
    print(f"Action Accuracy: %{accuracy*100:.2f}")
    return accuracy

# Örnek: gerçek veri dosyasını yükle
with open("21500001_last_result.json", "r") as f:
    real_data = json.load(f)

# Eğer aksiyonlar string ise sayısala çevir:
action_map = {"pass": 0, "shot": 1, "defend": 2, "dribble": 3, "": 4}
for sample in real_data:
    if isinstance(sample[1], str):
        sample[1] = action_map.get(sample[1], 4)

# Accuracy hesapla
#evaluate_action_accuracy(dqn, real_data, device)


