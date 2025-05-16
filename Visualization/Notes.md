
---------------------- Karsilastirma metrikleri -----------------------
1 - Eğri İncelenmesi: Eğitimin her episode için ortalama ve medyan ödül değerleri
2 - Tum lig boyunca ortalama mac score'lari
3 - Korelasyon Analizleri: Öğrenilen model çıktıları ile gerçek maçlardaki istatistikler (örneğin asist sayısı, top çalma) arasında korelasyon 

----------------------- Gorsellestirme -------------------------------------
1 - HEAT MAP (bunlari hem gercek veri hemde modelin ciktisi ile karsilastirabiliriz)
    a. Oyuncu Hareket Isı Haritası
    b. Şut Yoğunluğu Haritası
    c. Top Pase Isı Haritası:

2 - Reward Map: Saha üzerinde çeşitli pozisyonlardan oynandığında oduller cubuk grafiginde gosterilebilir.

3 - Ağ Analizleri (Network Graphs)
    a.Pas Ağı: Oyuncular arası pas sayılarını düğümler ve kenarlar olarak grafik üzerinde gösterin. Düğüm büyüklüğü pas sayısını, kenar kalınlığı pas sıklığını yansıtabilir.

4 - Öğrenme Eğrisi (Learning Curve): Eğitim sırasında ortalama ödül, kayıp (loss) ve ε‑değeri gibi metrikleri zaman içinde çizgi grafiğinde gösterin. Modelin ne zaman “plateau” yaptığını net görürsünüz.

5 - Score Differential Zaman Serisi: Simülasyon maçlarında skor farkının (home – away) zamana göre değişimini birkaç örnek maç için üst üste veya küçük multiples halinde çizin.

6 - Q‑Değer Isı Haritaları: Sahadaki her konum için ajanınızın “action-value” (Q(s,a))’sını gösteren 2D ısı haritası. Hangi bölgelerden hangi hamlelerin daha değerli olduğunu görürsünüz.

7 - Şut Açı Dağılımı: Potaya göre açıya göre atılan şutların frekansını polar histogramda (rose plot) veya düz histogramda sunun.


---- anilz et bunlari okumadim. -----
3. Hareket ve Oyun Akışı Animasyonları
Frame‑by‑Frame Animasyon:

Gerçek bir hücum sekansını JSON verinizden alıp Matplotlib/Ffmpeg ile video veya GIF oluşturun. Top ve oyuncu izleri bir animasyonda net görünür.

Flow Field (Akış Alanı):

Saha üzerindeki küçük “v” oklarıyla ortalama top ve oyuncu hız vektörlerini gösterin; stratejinin genel akışını görselleştirir.

4. Karşılaştırmalı Radar ve Paralel Koordinatlar
Radar Grafiği:

Farklı modellerin veya farklı hyper‑parametre ayarlarının “kazanma oranı, ortalama ödül, kritik an performansı, varyans, işlem süresi” gibi beş-altı boyutunu tek bir radar grafiğinde karşılaştırın.

Paralel Koordinatlar:

Ablasyon deneyleriniz (örneğin state özelliği çıkarma) sonuçlarını paralel eksenlerde gösterip, hangi konfigürasyonun nerede iyi/kötü kaldığını ortaya koyun.

5. Ağ ve İlişki Diyagramları
Chord Diagram (Pas Diyagramı):

Daire etrafında oyuncu düğümleri, oyuncular arası pasları yayılan kirişlerle gösterin—takım içi bağlantıları ve pas yoğunluğunu vurgular.

Heatmap of Passing Matrix:

Oyuncular arası pas sayılarının satır‑sütun ısı haritası: kimden kime, ne kadar pas gitmiş?

6. Boyutsal İndirgeme ve Kümeleme
t-SNE / UMAP Görselleştirmesi:

Öğrenilen state‑embedding’leri veya Q‑vektörlerini 2D’ye indirgenmiş olarak scatter plot’ta gösterin; benzer oyun durumlarının nasıl kümelendiğini gözlemleyin.

Kümeleme Sonuçları:

Saha pozisyonlarına göre “attack”, “defence” gibi otomatik küme etiketlerini kullanıp farklı kümeleri renklendirin.

7. Kritik An ve Sensitivite Analizleri
Perturbation Sensitivity Map:

Sahadaki konumlara küçük gürültü ekleyip ödül değişimini ısı haritasında gösterin; modelin nerelerde daha kararlı/narin olduğunu ortaya çıkarın.

Counterfactual Analysis:

Belirli bir durumda (ör. hücum seti) “eğer bu pas yerine şut çekseydi?” diyerek Q‑değer farklarını çubuk grafikle sunun.

8. KPI Dashboard Örneği
Interaktif Paneller (Statik Örnek Görsel):

Raporunuza ekleyebileceğiniz örnek bir “dashboard” tasarımı:

Üstte genel metrikler (win %, avg. reward)

Ortada learning curve

Altta heatmap ve polar shot chart

Böylece okuyucu tüm kritik metriklere tek bakışta hakim olur.
