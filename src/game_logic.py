import numpy as np
import pyttsx3

class GameLogic:
    def __init__(self, ruleset, player_names = ['Player 1', 'Player 2'], x01=501, num_legs=1, call_scores=True):
        print("game_logic.py: __init__\n")
        self.ruleset = ruleset
        self.x01 = x01
        self.num_legs = num_legs
        self.player_names = player_names
        self.num_players = len(self.player_names)
        self.leg_scores = [0] * self.num_players
        self.scores = [x01] * self.num_players
        self.starting_player, self.current_player = 0, 0
        #self.point_history = {i: [[] for _ in range(self.num_legs*2 -1)] for i in range(self.num_players)}
        self.num_dart_history = np.zeros((self.num_players, self.num_legs*2 - 1))
        self.num_visits_history = np.zeros((self.num_players, self.num_legs*2 - 1))
        self.averages = np.zeros(self.num_players)

        self.call_scores = call_scores
        if self.call_scores:
            self.text_to_speech = pyttsx3.init()


    def get_score_for_dart(self, dart):
        print("game_logic.py: get_score_for_dart\n")
        if dart == 'DB':
            return 50
        elif dart == 'SB':
            return 25
        elif dart == 'miss':
            return 0
        else:
            number = int(dart[1:])
            if dart[0] == 'S':
                return number
            elif dart[0] == 'T':
                return number*3
            elif dart[0] == 'D':
                return number*2
            
        
    
    