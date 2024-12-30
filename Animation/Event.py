from Constant import Constant
from Moment import Moment
from Team import Team
from Ball import Ball
from Player import Player
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle, Arc
import matplotlib.widgets as widgets
from matplotlib.widgets import Button
import pandas as pd
from matplotlib.widgets import TextBox


'''
[ ] faul kismini dusun 
[ ] similasyon renklendirme 
[ ] REPLAY mekanizmasi => bir seri veriyi alip gerisini kaydediyor (DQN arastir bu kismi)
[ ] ardisik verileri eleme yap
[ ] veriseti website incele
[ ] websitesi yap mac ve sut istatisleri gibi 
 '''



class Event:
    """A class for handling and showing events"""

    def __init__(self, event):
        moments = event['moments']
        self.moments = [Moment(moment) for moment in moments]
        home_players = event['home']['players']
        guest_players = event['visitor']['players']
        players = home_players + guest_players
        player_ids = [player['playerid'] for player in players]
        player_names = [" ".join([player['firstname'], player['lastname']]) for player in players]
        player_jerseys = [player['jersey'] for player in players]
        values = list(zip(player_names, player_jerseys))
        self.player_ids_dict = dict(zip(player_ids, values))
        self.current_index = 0 

    def update_forward_frame(self, event):
        """İleri hareket için fonksiyon"""
        if self.current_index < len(self.moments) - 1:
            self.current_index += 1
        self.update_plot()

    def update_backward_frame(self, event):
        """Geri hareket için fonksiyon"""
        if self.current_index > 0:
            self.current_index -= 1
        self.update_plot()

    def apply_event(self, event):
        """Kullanıcının girdiği event numarasını işle ve animasyonu başlat."""
        try:
            new_event_index = int(self.event_input.text)
            if 0 <= new_event_index < len(self.moments):
                self.current_index = new_event_index
                self.update_plot()
            else:
                print(f"Hatalı giriş! Geçerli bir event değeri girin (0 ile {len(self.moments) - 1} arasında).")
        except ValueError:
            print("Lütfen geçerli bir tam sayı girin.")
    
    def apply_selected_event(self, event):
        """Apply the event from the event number input box."""
        try:
            selected_event_number = int(self.event_no_input.text)
            if 0 <= selected_event_number < len(self.moments):
                self.current_index = selected_event_number
                self.update_plot()
            else:
                print(f"Invalid event number! Please enter a value between 0 and {len(self.moments) - 1}.")
        except ValueError:
            print("Please enter a valid integer.")

    def update_plot(self):
        moment = self.moments[self.current_index]
        for j, circle in enumerate(self.player_circles):
            circle.center = moment.players[j].x, moment.players[j].y
            self.annotations[j].set_position(circle.center)
        self.ball_circle.center = moment.ball.x, moment.ball.y
        self.ball_circle.radius = moment.ball.radius / Constant.NORMALIZATION_COEF
        clock_text = 'Quarter {:d}\n{:02d}:{:02d}\n{:03.1f}'.format(
            moment.quarter,
            int(moment.game_clock) % 3600 // 60,
            int(moment.game_clock) % 60,
            moment.shot_clock
        )
        self.clock_info.set_text(clock_text)
        plt.draw()

    def show(self):
        ax = plt.axes(xlim=(Constant.X_MIN, Constant.X_MAX), ylim=(Constant.Y_MIN, Constant.Y_MAX))
        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)

        # Court background
        court = plt.imread("court.png")
        plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN])

        # TextBox for Moment No input
        moment_box_ax = plt.axes([0.2, 0.01, 0.3, 0.05])  # (x, y, genişlik, yükseklik)
        self.event_input = TextBox(moment_box_ax, 'Moment No: ', initial=str(self.current_index))

        # TextBox for Event No input
        event_box_ax = plt.axes([0.2, 0.08, 0.3, 0.05])  # (x, y, genişlik, yükseklik)
        self.event_no_input = TextBox(event_box_ax, 'Event No: ', initial="")

        # Apply Button for Moment No
        apply_button_ax = plt.axes([0.55, 0.01, 0.1, 0.05])  # (x, y, genişlik, yükseklik)
        apply_button = Button(apply_button_ax, 'Apply Moment')
        apply_button.on_clicked(self.apply_event)

        # Apply Button for Event No
        apply_event_button_ax = plt.axes([0.55, 0.08, 0.1, 0.05])  # (x, y, genişlik, yükseklik)
        apply_event_button = Button(apply_event_button_ax, 'Apply Event')
        apply_event_button.on_clicked(self.apply_selected_event)

        # Forward and Backward Buttons
        forward_button_ax = plt.axes([0.8, 0.01, 0.1, 0.05])  # (x, y, genişlik, yükseklik)
        forward_button = Button(forward_button_ax, 'İleri')
        forward_button.on_clicked(self.update_forward_frame)

        backward_button_ax = plt.axes([0.8, 0.07, 0.1, 0.05])  # (x, y, genişlik, yükseklik)
        backward_button = Button(backward_button_ax, 'Geri')
        backward_button.on_clicked(self.update_backward_frame)

        # Annotations and Circles
        start_moment = self.moments[self.current_index]
        self.clock_info = ax.annotate('', xy=[Constant.X_CENTER, Constant.Y_CENTER],
                                      color='black', horizontalalignment='center',
                                      verticalalignment='center')
        self.annotations = [ax.annotate(self.player_ids_dict[player.id][1], xy=[0, 0], color='w',
                                        horizontalalignment='center', verticalalignment='center', fontweight='bold')
                            for player in start_moment.players]
        self.player_circles = [plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=player.color)
                               for player in start_moment.players]
        self.ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=start_moment.ball.color)

        for circle in self.player_circles:
            ax.add_patch(circle)
        ax.add_patch(self.ball_circle)

        self.update_plot()  
        plt.show()

'''

from Constant import Constant
from Moment import Moment
from Team import Team
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle, Arc
import matplotlib.widgets as widgets

class Event:
    """A class for handling and showing events"""

    def _init_(self, event):
        moments = event['moments']
        self.moments = [Moment(moment) for moment in moments]
        self.current_moment = 0  # Start at the first moment
        home_players = event['home']['players']
        guest_players = event['visitor']['players']
        players = home_players + guest_players
        player_ids = [player['playerid'] for player in players]
        player_names = [" ".join([player['firstname'], player['lastname']]) for player in players]
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
        """Move to the next moment"""
        if self.current_moment < len(self.moments) - 1:
            self.current_moment += 1
        event()

    def previous_moment(self, event):
        """Move to the previous moment"""
        if self.current_moment > 0:
            self.current_moment -= 1
        event()

    def show(self):
        # Create plot
        ax = plt.axes(xlim=(Constant.X_MIN, Constant.X_MAX), ylim=(Constant.Y_MIN, Constant.Y_MAX))
        ax.axis('off')
        fig = plt.gcf()
        ax.grid(False)
        start_moment = self.moments[self.current_moment]
        player_dict = self.player_ids_dict

        clock_info = ax.annotate('', xy=[Constant.X_CENTER, Constant.Y_CENTER],
                                 color='black', horizontalalignment='center',
                                 verticalalignment='center')

        annotations = [ax.annotate(self.player_ids_dict[player.id][1], xy=[0, 0], color='w',
                                   horizontalalignment='center',
                                   verticalalignment='center', fontweight='bold')
                       for player in start_moment.players]

        # Prepare table (same as before)
        sorted_players = sorted(start_moment.players, key=lambda player: player.team.id)
        
        home_player = sorted_players[0]
        guest_player = sorted_players[5]
        column_labels = tuple([home_player.team.name, guest_player.team.name])
        column_colours = tuple([home_player.team.color, guest_player.team.color])
        cell_colours = [column_colours for _ in range(5)]
        
        home_players = [' #'.join([player_dict[player.id][0], player_dict[player.id][1]]) for player in sorted_players[:5]]
        guest_players = [' #'.join([player_dict[player.id][0], player_dict[player.id][1]]) for player in sorted_players[5:]]
        players_data = list(zip(home_players, guest_players))

        table = plt.table(cellText=players_data,
                              colLabels=column_labels,
                              colColours=column_colours,
                              colWidths=[Constant.COL_WIDTH, Constant.COL_WIDTH],
                              loc='bottom',
                              cellColours=cell_colours,
                              fontsize=Constant.FONTSIZE,
                              cellLoc='center')
        table.scale(1, Constant.SCALE)
        table_cells = table.get_celld().values()
        for cell in table_cells:
            cell._text.set_color('white')

        player_circles = [plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=player.color)
                          for player in start_moment.players]
        ball_circle = plt.Circle((0, 0), Constant.PLAYER_CIRCLE_SIZE, color=start_moment.ball.color)
        for circle in player_circles:
            ax.add_patch(circle)
        ax.add_patch(ball_circle)

        anim = animation.FuncAnimation(
                         fig, self.update_radius,
                         fargs=(player_circles, ball_circle, annotations, clock_info),
                         frames=len(self.moments), interval=Constant.INTERVAL)

        # Court background image
        court = plt.imread("court.png")
        plt.imshow(court, zorder=0, extent=[Constant.X_MIN, Constant.X_MAX - Constant.DIFF,
                                            Constant.Y_MAX, Constant.Y_MIN])

        # Button for Next Moment
        ax_next = plt.axes([0.8, 0.01, 0.1, 0.075])
        btn_next = widgets.Button(ax_next, 'Next')
        btn_next.on_clicked(lambda x: self.next_moment(anim.event_source.stop))

        # Button for Previous Moment
        ax_prev = plt.axes([0.7, 0.01, 0.1, 0.075])
        btn_prev = widgets.Button(ax_prev, 'Previous')
        btn_prev.on_clicked(lambda x: self.previous_moment(anim.event_source.stop))

        plt.show() '''