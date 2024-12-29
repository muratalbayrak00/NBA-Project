import json
import numpy as np
import logging

class BasketballEnvironment:
    def __init__(self, json_file_path, home_basket_coords, visitor_basket_coords):
        self.json_file_path = json_file_path
        self.home_basket_coords = home_basket_coords
        self.visitor_basket_coords = visitor_basket_coords
        self.match_data = []  # match_data özelliğini burada tanımlıyoruz
        self.reset()

    def reset(self):
        """
        Oyunu sıfırlar ve başlangıç state'ini döndürür.
        """
        with open(self.json_file_path, "r") as file:
            self.data = json.load(file)
        
        self.events = self.data["events"]
        self.event_idx = 0
        self.moment_idx = 1
        self.state, self.done = self.process_moment()
        return self.state
    
    def identify_ball_holder(self, current_moment):
        ball_position = current_moment[5][0][2:4]
        player = np.array([
                [player[1], player[2], player[3]] for player in current_moment[5]
            ])
        player = np.delete(player, 0, axis=0)  # Topu çıkar
        min_distance = 90
        min_id = 0
        for pos in player:
            player_position = pos[1:]
            distance = np.linalg.norm(player_position - ball_position)
            if (distance < min_distance):
                min_distance = distance
                min_id = pos[0]
        min_id = int(min_id)
        if (min_distance < 2):
            return min_id
        else:
            return 0

    def save_match_data(self, state, action, reward):
        # Veriyi uygun formatta hazırlayıp listeye ekle
        self.match_data.append([state.tolist(), action, reward])
        
        # JSON dosyasına kaydet
        try:
            with open('match_data.json', 'w') as f:
                json.dump(self.match_data, f)
        except Exception as e:
            logging.error(f"Error saving match data: {e}")

    def process_moment(self):
        """
        JSON verisinden tüm eventleri ve momentleri işleyerek state'leri döndürür.
        """
        try:
            # Döngü ile tüm eventleri ve momentleri gez
            while self.event_idx < len(self.events):
                event = self.events[self.event_idx]
                moments = event["moments"]

                # Her bir moment için işleme yap
                while self.moment_idx < len(moments):
                    current_moment = moments[self.moment_idx]

                    # Oyuncu konumlarını ve top konumunu al
                    player_positions = np.array([
                        [player[2], player[3]] for player in current_moment[5]
                    ])
                    player_positions = np.delete(player_positions, 0, axis=0)  # Topu çıkar
                    ball_position = current_moment[5][0][2:]  # Topun koordinatları
                    time_remaining = current_moment[3]  # Zaman bilgisi

                    # Topu tutan oyuncuyu bul
                    ball_holder = self.identify_ball_holder(current_moment)

                    # State'i oluştur
                    state = np.concatenate((player_positions.flatten(), ball_position, [time_remaining, ball_holder]))
                    np.set_printoptions(suppress=True)

                    # Eğer state beklenen boyutta değilse hata ver
                    # State'in boyutunu kontrol et
                    if len(player_positions.flatten()) + len(ball_position) + 2 != 25:
                        logging.warning(f"Unexpected state length at Event {self.event_idx}, Moment {self.moment_idx}")
                        # Eksik veri durumunda 0'lar ile tamamlayabiliriz
                        while len(state) < 25:
                            state = np.append(state, 0)


                    # Varsayılan action ve reward değerleri
                    action = 0  # Bu değer, gerçek uygulamanıza göre belirlenmelidir
                    reward = 0  # Bu değer, gerçek uygulamanıza göre belirlenmelidir
                    """
                    # Bir sonraki momentin state'ini elde et
                    if self.moment_idx + 1 < len(moments):
                        next_state = np.concatenate((
                            np.array([
                                [player[2], player[3]] for player in moments[self.moment_idx + 1][5]
                            ]).flatten(), 
                            moments[self.moment_idx + 1][5][0][2:], 
                            [moments[self.moment_idx + 1][3], ball_holder]
                        ))
                    else:
                        next_state = state  # Eğer sonraki moment yoksa mevcut state'i al
                    """
                    # Match verilerini dosyaya kaydet
                    self.save_match_data(state, action, reward)

                    # Bir sonraki momenti işlemek için ilerle
                    self.moment_idx += 1

                # Eğer event'teki tüm momentler bitmişse bir sonraki eventi işle
                self.event_idx += 1
                self.moment_idx = 1  # Event değiştiği için moment sıfırlanır

            return None, True  # Tüm eventler bittiğinde sonlandır

        except Exception as e:
            logging.error(f"Error processing moment: {e}")
            return None, True


# Çevreyi başlatma
env = BasketballEnvironment("fdni0021500491.json", [5.37, 24.7], [88, 24.7])
