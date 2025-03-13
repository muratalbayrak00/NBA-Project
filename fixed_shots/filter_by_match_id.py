import pandas as pd

# Dosya yolu
file_path = "fixed_shots/shots_fixed.csv"

# CSV dosyasını oku
df = pd.read_csv(file_path, dtype={"GAME_ID": str})  # GAME_ID'nin string olarak okunmasını sağla

# Filtreleme yap
game_id_to_filter = "0021500491"  # İstediğiniz GAME_ID değerini buraya yazın
filtered_df = df[df["GAME_ID"] == game_id_to_filter]

# Sonuçları yeni bir CSV dosyasına kaydet
output_path = "fixed_shots/filtered_game_0021500491.csv"
filtered_df.to_csv(output_path, index=False)

print(f"Filtrelenmiş veriler {output_path} dosyasına kaydedildi.")


'''
ACTION_TYPE	Yapılan hareketin türü (örneğin: "Jump Shot", "Dunk", "Layup")
EVENTTIME	Olayın maç içindeki toplam süresi (milisaniye cinsinden olabilir)
EVENT_TYPE	Olayın türü (örneğin: şut, faul, mola vb.)
GAME_DATE	Maçın oynandığı tarih (YYYY-MM-DD formatında)
GAME_EVENT_ID	Maç içindeki olayın benzersiz kimliği
GAME_ID	Maçın kimliği (NBA tarafından belirlenen eşsiz maç kodu)
GRID_TYPE	Konum verisinin türü (şut haritası için olabilir)
HTM	Ev sahibi takımın NBA kısaltması (örn: "LAL" = Los Angeles Lakers)
LOC_X	Şutun atıldığı X koordinatı (saha içindeki yatay konum)
LOC_Y	Şutun atıldığı Y koordinatı (saha içindeki dikey konum)
MINUTES_REMAINING	O çeyrekte kalan dakika sayısı
PERIOD	Maçın hangi periyodunda olduğu (1, 2, 3, 4 veya uzatma periyotları)
PLAYER_ID	Şutu atan oyuncunun NBA kimlik numarası
PLAYER_NAME	Oyuncunun adı ve soyadı
QUARTER	Çeyrek bilgisi (Genellikle PERİOD ile aynı)
SECONDS_REMAINING	O çeyrekte kalan saniye sayısı
SHOT_ATTEMPTED_FLAG	Şut atılıp atılmadığını gösterir (1 = Şut denendi, 0 = Şut denemesi yok)
SHOT_DISTANCE	Şutun potaya olan mesafesi (fit cinsinden)
SHOT_MADE_FLAG	Şutun isabetli olup olmadığı (1 = İsabetli, 0 = Kaçırdı)
SHOT_TIME	Şutun atıldığı toplam süre (maç içindeki genel zaman)
SHOT_TYPE	Şut türü (örn: "2PT Field Goal" veya "3PT Field Goal")
SHOT_ZONE_AREA	Şutun atıldığı saha bölgesi (örneğin: "Sağ Kanat", "Merkez")
SHOT_ZONE_BASIC	Şutun temel kategorisi (örneğin: "Atış", "Turnike", "Smaç")
SHOT_ZONE_RANGE	Şutun menzili (örneğin: "Boyalı Alan", "Orta Mesafe", "Üç Sayılık Bölge")
TEAM_ID	Şutu atan oyuncunun takım kimlik numarası
TEAM_NAME	Oyuncunun takımı (örn: "Golden State Warriors")
VTM	Deplasman takımının NBA kısaltması (örn: "GSW" = Golden State Warriors)

'''