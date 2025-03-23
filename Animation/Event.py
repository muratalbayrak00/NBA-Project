from Constant import Constant
from Moment import Moment
from Team import Team
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle, Arc

from matplotlib.widgets import Button

class Event:
    """A class for handling and showing events"""

    def __init__(self, event):
        moments = event['moments']
        self.moments = [Moment(moment) for moment in moments]
        self.current_moment_index = 0
        home_players = event['home']['players']
        guest_players = event['visitor']['players']
        players = home_players + guest_players
        player_ids = [player['playerid'] for player in players]
        player_names = [" ".join([player['firstname'],
                        player['lastname']]) for player in players]
        player_jerseys = [player['jersey'] for player in players]
        values = list(zip(player_names, player_jerseys))
        self.player_ids_dict = dict(zip(player_ids, values))

    def update_radius(self, i, player_circles, ball_circle, annotations, clock_info):
        moment = self.moments[i]
        for j, circle in enumerate(player_circles):
            circle.center = moment.players[j].x, moment.players[j].y
            annotations[j].set_position(circle.center)
            clock_test = 'Quarter {:d}\n {:02d}:{:02d}\n {:03.1f}'.format(
                         moment.quarter,
                         int(moment.game_clock) % 3600 // 60,
                         int(moment.game_clock) % 60,
                         moment.shot_clock)
            clock_info.set_text(clock_test)
        ball_circle.center = moment.ball.x, moment.ball.y
        ball_circle.radius = moment.ball.radius / Constant.NORMALIZATION_COEF
        return player_circles, ball_circle

    def next_moment(self, event):
        if self.current_moment_index < len(self.moments) - 1:
            self.current_moment_index += 1
            self.update_radius(self.current_moment_index, self.player_circles, self.ball_circle, self.annotations, self.clock_info)
            plt.draw()

    def prev_moment(self, event):
        if self.current_moment_index > 0:
            self.current_moment_index -= 1
            self.update_radius(self.current_moment_index, self.player_circles, self.ball_circle, self.annotations, self.clock_info)
            plt.draw()

    def show(self):
        # Figür boyutunu küçült (genişlik, yükseklik)
        fig, ax = plt.subplots(figsize=(8, 6))  # Boyutu küçülttük (12, 10 yerine 8, 6)
        ax.set_xlim(Constant.X_MIN, Constant.X_MAX)
        ax.set_ylim(Constant.Y_MIN, Constant.Y_MAX)
        ax.axis('off')
        ax.grid(False)

        # Clock bilgisini ekle
        self.clock_info = ax.annotate('', xy=[Constant.X_CENTER, Constant.Y_CENTER],
                                    color='black', horizontalalignment='center',
                                    verticalalignment='center')

        # Oyuncu isimlerini ve numaralarını ekle
        self.annotations = [ax.annotate(self.player_ids_dict[player.id][1], xy=[0, 0], color='w',
                                    horizontalalignment='center',
                                    verticalalignment='center', fontweight='bold')
                        for player in self.moments[0].players]

        # Tablo verilerini hazırla
        sorted_players = sorted(self.moments[0].players, key=lambda player: player.team.id)
        home_player = sorted_players[0]
        guest_player = sorted_players[5]
        column_labels = tuple([home_player.team.name, guest_player.team.name])
        column_colours = tuple([home_player.team.color, guest_player.team.color])
        cell_colours = [column_colours for _ in range(5)]
        
        home_players = [' #'.join([self.player_ids_dict[player.id][0], self.player_ids_dict[player.id][1]]) 
                    for player in sorted_players[:5]]
        guest_players = [' #'.join([self.player_ids_dict[player.id][0], self.player_ids_dict[player.id][1]]) 
                        for player in sorted_players[5:]]
        players_data = list(zip(home_players, guest_players))

        # Tablo oluştur ve ekrana ekle
        table = plt.table(cellText=players_data,
                        colLabels=column_labels,
                        colColours=column_colours,
                        colWidths=[0.25, 0.25],  # Sütun genişliklerini artır
                        loc='bottom',
                        bbox=[0.1, -0.3, 0.8, 0.25],  # Tablo konumu ve boyutu (x, y, width, height)
                        cellColours=cell_colours,
                        fontsize=10,  # Font boyutunu küçült
                        cellLoc='center')
        table.scale(1, 1.5)  # Tablo boyutunu ayarla
        table_cells = table.get_children()
        for cell in table_cells:
            cell._text.set_color('white')  # Metin rengini beyaz yap

        # Oyuncu ve top çemberlerini ekle
        self.player_circles = [plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=player.color)
                            for player in self.moments[0].players]
        self.ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE,
                                    color=self.moments[0].ball.color)
        for circle in self.player_circles:
            ax.add_patch(circle)
        ax.add_patch(self.ball_circle)

        # Butonları ekle
        ax_prev = plt.axes([0.7, 0.02, 0.1, 0.05])  # Buton konumu
        ax_next = plt.axes([0.81, 0.02, 0.1, 0.05])  # Buton konumu
        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')
        btn_prev.on_clicked(self.prev_moment)
        btn_next.on_clicked(self.next_moment)

        # Court resmini ekle
        court = plt.imread("court.png")
        ax.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX, Constant.Y_MIN, Constant.Y_MAX])

        plt.show()