import json

home_player_ids = {2547,2548,2405,201609,204020,2736,2365,101123,2757,1626159,202355,2617}
visitor_player_ids = {101109,101111,200826,203939,101114,201580,203098,202379,202083,202718,2585,2734,1717}
def load_match_data(file_path):
    """Load match data from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def parse_state(state):
    """Parse state into meaningful components."""
    players_home = [(state[i], state[i+1]) for i in range(0, 10, 2)]  # Home team players' positions
    players_away = [(state[i], state[i+1]) for i in range(10, 20, 2)]  # Away team players' positions
    ball_position = (state[20], state[21], state[22])  # Ball position (x, y, z)
    remaining_time = state[23]  # Time remaining in the attack
    ball_owner = state[24]  # Player ID who owns the ball
    return players_home, players_away, ball_position, remaining_time, ball_owner

def detect_action(state1, state2):
    """Detect the action that occurred between two states."""
    players_home1, players_away1, ball_position1, time1, ball_owner1 = parse_state(state1)
    players_home2, players_away2, ball_position2, time2, ball_owner2 = parse_state(state2)
    home_basket_coords = (88.8, 24.7)  # Ev sahibi takım sepet koordinatları (örnek)
    visitor_basket_coords = (5.37, 24.7)  # Rakip takım sepet koordinatları (örnek)

    actions = []   #actions = ["pass", "shoot", "dribble", "defend", "idle"]
    rewards = []



    if ball_owner1 in visitor_player_ids:  # Top rakipte
            #print("defend")
            actions.append("defend")
            if ball_owner2 in visitor_player_ids:
                #print("defend_fail")
                rewards.append("defend_fail")
            elif ball_owner2 in home_player_ids:
                rewards.append("steal")
    elif ball_owner1 in home_player_ids:  # Top ev sahibinde
        # pass, shot, dribble
        if ball_owner1 == ball_owner2: # dribble
            actions.append("dribble")
        else:  # pass or shot    11 feet i geçsin
            # Topun hareket yönü vektörü
            motion_vector = (ball_position2[0] - ball_position1[0], ball_position2[1] - ball_position1[1])

            # Pota ile olan doğrultuyu hesapla
            basket_vector = (home_basket_coords[0] - ball_position2[0], home_basket_coords[1] - ball_position2[1])

            # Vektörlerin noktasal çarpımını kullanarak hareketin potaya doğru olup olmadığını kontrol et
            dot_product = motion_vector[0] * basket_vector[0] + motion_vector[1] * basket_vector[1]

            # Eğer noktasal çarpım pozitifse, hareket potaya doğru yöneliyor demektir
            if dot_product > 0 and ball_position2[2] > 6.5:  # Top havada ve potaya doğru yöneliyorsa
                actions.append("shoot")
            else:
                actions.append("pass")

        
    else: # Sahipsiz top
        actions.append("idle")


    return actions

def main():
    # Load match data
    match_data = load_match_data('match_data.json')

    for i in range(len(match_data) - 1):
        state1 = match_data[i][0]
        state2 = match_data[i + 1][0]
        actions = detect_action(state1, state2)
        print(f"Actions between state {i} and state {i+1}: {actions}")

if __name__ == "__main__":
    main()
