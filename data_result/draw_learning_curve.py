import matplotlib.pyplot as plt
import re

# Log verisi (sadece kısa bir örnek yazdım, sen kendi logunun tamamını buraya yapıştırabilirsin)
log_text_1000 = """
Episode 1/1000, Reward: -605.00, Epsilon Home: 0.1492,Epsilon Away: 0.3483, Score: 21-24
Episode 11/1000, Reward: -359.00, Epsilon Home: 0.1420,Epsilon Away: 0.3312, Score: 38-0
Episode 21/1000, Reward: -342.00, Epsilon Home: 0.1350,Epsilon Away: 0.3150, Score: 15-2
Episode 31/1000, Reward: -618.00, Epsilon Home: 0.1284,Epsilon Away: 0.2996, Score: 73-7
Episode 41/1000, Reward: -625.00, Epsilon Home: 0.1221,Epsilon Away: 0.2850, Score: 82-0
Episode 51/1000, Reward: -442.00, Epsilon Home: 0.1162,Epsilon Away: 0.2710, Score: 19-15
Episode 61/1000, Reward: -378.00, Epsilon Home: 0.1105,Epsilon Away: 0.2578, Score: 38-7
Episode 71/1000, Reward: -270.00, Epsilon Home: 0.1051,Epsilon Away: 0.2452, Score: 23-4
Episode 81/1000, Reward: -285.00, Epsilon Home: 0.0999,Epsilon Away: 0.2332, Score: 15-8
Episode 91/1000, Reward: -288.00, Epsilon Home: 0.0951,Epsilon Away: 0.2218, Score: 10-9
Episode 101/1000, Reward: -469.00, Epsilon Home: 0.0904,Epsilon Away: 0.2110, Score: 26-20
Episode 111/1000, Reward: -471.00, Epsilon Home: 0.0860,Epsilon Away: 0.2006, Score: 0-4
Episode 121/1000, Reward: -407.00, Epsilon Home: 0.0818,Epsilon Away: 0.1908, Score: 23-9
Episode 131/1000, Reward: -424.00, Epsilon Home: 0.0778,Epsilon Away: 0.1815, Score: 29-20
Episode 141/1000, Reward: -429.00, Epsilon Home: 0.0740,Epsilon Away: 0.1726, Score: 42-18
Episode 151/1000, Reward: -437.00, Epsilon Home: 0.0704,Epsilon Away: 0.1642, Score: 34-16
Episode 161/1000, Reward: -270.00, Epsilon Home: 0.0669,Epsilon Away: 0.1562, Score: 14-2
Episode 171/1000, Reward: -415.00, Epsilon Home: 0.0637,Epsilon Away: 0.1485, Score: 34-11
Episode 181/1000, Reward: -560.00, Epsilon Home: 0.0605,Epsilon Away: 0.1413, Score: 36-11
Episode 191/1000, Reward: -288.00, Epsilon Home: 0.0576,Epsilon Away: 0.1344, Score: 32-10
Episode 201/1000, Reward: -726.00, Epsilon Home: 0.0548,Epsilon Away: 0.1278, Score: 84-40
Episode 211/1000, Reward: -759.00, Epsilon Home: 0.0521,Epsilon Away: 0.1215, Score: 134-0
Episode 221/1000, Reward: -869.00, Epsilon Home: 0.0495,Epsilon Away: 0.1156, Score: 122-35
Episode 231/1000, Reward: -756.00, Epsilon Home: 0.0471,Epsilon Away: 0.1100, Score: 135-0
Episode 241/1000, Reward: -777.00, Epsilon Home: 0.0448,Epsilon Away: 0.1046, Score: 127-5
Episode 251/1000, Reward: -878.00, Epsilon Home: 0.0426,Epsilon Away: 0.0995, Score: 81-28
Episode 261/1000, Reward: -760.00, Epsilon Home: 0.0405,Epsilon Away: 0.0946, Score: 156-0
Episode 271/1000, Reward: -221.00, Epsilon Home: 0.0386,Epsilon Away: 0.0900, Score: 9-2
Episode 281/1000, Reward: -339.00, Epsilon Home: 0.0367,Epsilon Away: 0.0856, Score: 19-22
Episode 291/1000, Reward: -385.00, Epsilon Home: 0.0349,Epsilon Away: 0.0814, Score: 34-26
Episode 301/1000, Reward: -477.00, Epsilon Home: 0.0332,Epsilon Away: 0.0774, Score: 29-18
Episode 311/1000, Reward: -389.00, Epsilon Home: 0.0316,Epsilon Away: 0.0736, Score: 45-10
Episode 321/1000, Reward: -428.00, Epsilon Home: 0.0300,Epsilon Away: 0.0700, Score: 18-18
Episode 331/1000, Reward: -391.00, Epsilon Home: 0.0285,Epsilon Away: 0.0666, Score: 27-5
Episode 341/1000, Reward: -483.00, Epsilon Home: 0.0271,Epsilon Away: 0.0633, Score: 20-23
Episode 351/1000, Reward: -422.00, Epsilon Home: 0.0258,Epsilon Away: 0.0603, Score: 19-24
Episode 361/1000, Reward: -527.00, Epsilon Home: 0.0246,Epsilon Away: 0.0573, Score: 30-10
Episode 371/1000, Reward: -432.00, Epsilon Home: 0.0234,Epsilon Away: 0.0545, Score: 37-17
Episode 381/1000, Reward: -404.00, Epsilon Home: 0.0222,Epsilon Away: 0.0518, Score: 29-28
Episode 391/1000, Reward: -600.00, Epsilon Home: 0.0211,Epsilon Away: 0.0493, Score: 102-12
Episode 401/1000, Reward: -414.00, Epsilon Home: 0.0201,Epsilon Away: 0.0469, Score: 33-13
Episode 411/1000, Reward: -437.00, Epsilon Home: 0.0191,Epsilon Away: 0.0446, Score: 43-21
Episode 421/1000, Reward: -443.00, Epsilon Home: 0.0182,Epsilon Away: 0.0424, Score: 56-26
Episode 431/1000, Reward: -433.00, Epsilon Home: 0.0173,Epsilon Away: 0.0403, Score: 37-23
Episode 441/1000, Reward: -428.00, Epsilon Home: 0.0164,Epsilon Away: 0.0384, Score: 26-12
Episode 451/1000, Reward: -403.00, Epsilon Home: 0.0156,Epsilon Away: 0.0365, Score: 28-21
Episode 461/1000, Reward: -446.00, Epsilon Home: 0.0149,Epsilon Away: 0.0347, Score: 21-16
Episode 471/1000, Reward: -432.00, Epsilon Home: 0.0142,Epsilon Away: 0.0330, Score: 37-14
Episode 481/1000, Reward: -420.00, Epsilon Home: 0.0135,Epsilon Away: 0.0314, Score: 26-14
Episode 491/1000, Reward: -449.00, Epsilon Home: 0.0128,Epsilon Away: 0.0299, Score: 38-0
Episode 501/1000, Reward: -458.00, Epsilon Home: 0.0122,Epsilon Away: 0.0284, Score: 32-15
Episode 511/1000, Reward: -453.00, Epsilon Home: 0.0116,Epsilon Away: 0.0270, Score: 27-17
Episode 521/1000, Reward: -394.00, Epsilon Home: 0.0110,Epsilon Away: 0.0257, Score: 14-16
Episode 531/1000, Reward: -386.00, Epsilon Home: 0.0105,Epsilon Away: 0.0244, Score: 22-11
Episode 541/1000, Reward: -400.00, Epsilon Home: 0.0100,Epsilon Away: 0.0232, Score: 31-10
Episode 551/1000, Reward: -435.00, Epsilon Home: 0.0100,Epsilon Away: 0.0221, Score: 51-17
Episode 561/1000, Reward: -465.00, Epsilon Home: 0.0100,Epsilon Away: 0.0210, Score: 34-26
Episode 571/1000, Reward: -401.00, Epsilon Home: 0.0100,Epsilon Away: 0.0200, Score: 33-16
Episode 581/1000, Reward: -454.00, Epsilon Home: 0.0100,Epsilon Away: 0.0190, Score: 24-30
Episode 591/1000, Reward: -413.00, Epsilon Home: 0.0100,Epsilon Away: 0.0181, Score: 37-17
Episode 601/1000, Reward: -406.00, Epsilon Home: 0.0100,Epsilon Away: 0.0172, Score: 35-16
Episode 611/1000, Reward: -427.00, Epsilon Home: 0.0100,Epsilon Away: 0.0164, Score: 26-25
Episode 621/1000, Reward: -407.00, Epsilon Home: 0.0100,Epsilon Away: 0.0156, Score: 24-13
Episode 631/1000, Reward: -384.00, Epsilon Home: 0.0100,Epsilon Away: 0.0148, Score: 20-4
Episode 641/1000, Reward: -393.00, Epsilon Home: 0.0100,Epsilon Away: 0.0141, Score: 31-22
Episode 651/1000, Reward: -409.00, Epsilon Home: 0.0100,Epsilon Away: 0.0134, Score: 42-10
Episode 661/1000, Reward: -365.00, Epsilon Home: 0.0100,Epsilon Away: 0.0127, Score: 19-19
Episode 671/1000, Reward: -484.00, Epsilon Home: 0.0100,Epsilon Away: 0.0121, Score: 33-12
Episode 681/1000, Reward: -317.00, Epsilon Home: 0.0100,Epsilon Away: 0.0115, Score: 42-18
Episode 691/1000, Reward: -388.00, Epsilon Home: 0.0100,Epsilon Away: 0.0110, Score: 23-22
Episode 701/1000, Reward: -337.00, Epsilon Home: 0.0100,Epsilon Away: 0.0104, Score: 22-20
Episode 711/1000, Reward: -435.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 26-11
Episode 721/1000, Reward: -447.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 20-11
Episode 731/1000, Reward: -372.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 46-17
Episode 741/1000, Reward: -353.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 28-9
Episode 751/1000, Reward: -321.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 14-26
Episode 761/1000, Reward: -301.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 13-26
Episode 771/1000, Reward: -356.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 23-6
Episode 781/1000, Reward: -364.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 37-6
Episode 791/1000, Reward: -294.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 20-12
Episode 801/1000, Reward: -364.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 28-21
Episode 811/1000, Reward: -280.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 19-8
Episode 821/1000, Reward: -261.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 23-14
Episode 831/1000, Reward: -276.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 10-9
Episode 841/1000, Reward: -302.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 15-28
Episode 851/1000, Reward: -315.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 17-6
Episode 861/1000, Reward: -263.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 16-6
Episode 871/1000, Reward: -429.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 32-16
Episode 881/1000, Reward: -320.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 8-8
Episode 891/1000, Reward: -263.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 7-25
Episode 901/1000, Reward: -317.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 32-15
Episode 911/1000, Reward: -250.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 11-19
Episode 921/1000, Reward: -353.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 22-7
Episode 931/1000, Reward: -279.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 33-10
Episode 941/1000, Reward: -525.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 38-16
Episode 951/1000, Reward: -358.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 18-16
Episode 961/1000, Reward: -402.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 28-12
Episode 971/1000, Reward: -215.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 13-5
Episode 981/1000, Reward: -346.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 19-13
Episode 991/1000, Reward: -139.00, Epsilon Home: 0.0100,Epsilon Away: 0.0100, Score: 16-14
Model başarıyla kaydedildi: dqn_model_1000.pth
"""
log_text_10 = """
Episode 1/10, Reward: -577.00, Epsilon Home: 0.1492,Epsilon Away: 0.3483, Score: 43-10
Episode 2/10, Reward: -434.00, Epsilon Home: 0.1485,Epsilon Away: 0.3465, Score: 20-0
Episode 3/10, Reward: -435.00, Epsilon Home: 0.1478,Epsilon Away: 0.3448, Score: 37-17
Episode 4/10, Reward: -400.00, Epsilon Home: 0.1470,Epsilon Away: 0.3431, Score: 48-20
Episode 5/10, Reward: -640.00, Epsilon Home: 0.1463,Epsilon Away: 0.3413, Score: 43-3
Episode 6/10, Reward: -391.00, Epsilon Home: 0.1456,Epsilon Away: 0.3396, Score: 23-4
Episode 7/10, Reward: -624.00, Epsilon Home: 0.1448,Epsilon Away: 0.3379, Score: 48-15
Episode 8/10, Reward: -664.00, Epsilon Home: 0.1441,Epsilon Away: 0.3362, Score: 72-7
Episode 9/10, Reward: -680.00, Epsilon Home: 0.1434,Epsilon Away: 0.3346, Score: 61-0
Episode 10/10, Reward: -578.00, Epsilon Home: 0.1427,Epsilon Away: 0.3329, Score: 62-15
Model başarıyla kaydedildi: dqn_model_10.pth
"""
log_text_100 = """
Episode 10/100, Reward: -423.00, Epsilon Home: 0.1427,Epsilon Away: 0.3329, Score: 24-9
Episode 20/100, Reward: -453.00, Epsilon Home: 0.1357,Epsilon Away: 0.3166, Score: 45-2
Episode 30/100, Reward: -520.00, Epsilon Home: 0.1291,Epsilon Away: 0.3011, Score: 48-10
Episode 40/100, Reward: -537.00, Epsilon Home: 0.1227,Epsilon Away: 0.2864, Score: 65-0
Episode 50/100, Reward: -576.00, Epsilon Home: 0.1167,Epsilon Away: 0.2724, Score: 49-2
Episode 60/100, Reward: -358.00, Epsilon Home: 0.1110,Epsilon Away: 0.2591, Score: 38-20
Episode 70/100, Reward: -347.00, Epsilon Home: 0.1056,Epsilon Away: 0.2464, Score: 37-10
Episode 80/100, Reward: -483.00, Epsilon Home: 0.1004,Epsilon Away: 0.2344, Score: 58-0
Episode 90/100, Reward: -468.00, Epsilon Home: 0.0955,Epsilon Away: 0.2229, Score: 23-12
Episode 100/100, Reward: -375.00, Epsilon Home: 0.0909,Epsilon Away: 0.2120, Score: 16-19
Model başarıyla kaydedildi: dqn_model_100.pth
"""
# Episode ve Reward'ları çekmek için regex
episodes = []
rewards = []

for match in re.finditer(r"Episode (\d+)/\d+.*?Reward: (-?\d+\.\d+)", log_text_100):
    episode = int(match.group(1))
    reward = float(match.group(2))
    episodes.append(episode)
    rewards.append(reward)

# Grafik oluşturma
plt.figure(figsize=(12, 6))
plt.plot(episodes, rewards, label='Reward per Episode', color='blue', marker='o', markersize=4, linewidth=1)

# Grafik başlığı ve etiketler
plt.title('Learning Curve: Episode vs Reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid(True)
plt.legend()

# Göster
plt.tight_layout()
plt.show()
