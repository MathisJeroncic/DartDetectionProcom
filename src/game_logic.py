import numpy as np
import pyttsx3

class GameLogic:
    def __init__(self, ddartboard,ruleset, player_names = ['Player 1', 'Player 2'], x01=501, num_legs=1, call_scores=True):
        """
        Inputs:
            ruleset (object):
                Game rules handler defining scoring and validation logic.
            player_names (list[str], optional):
                Names of the players. Default is ['Player 1', 'Player 2'].
            x01 (int, optional):
                Starting score for each player (e.g., 501 or 301).
            num_legs (int, optional):
                Number of legs to be played.
            call_scores (bool, optional):
                Enables text-to-speech score announcements if True.

        Returns:
            None

        Purpose:
            Initializes the dart game logic state, including players,
            scores, statistics tracking, and optional text-to-speech
            for announcing scores.
        """

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
        self.dartboard=ddartboard

    def read_score(self, score):
        if self.call_scores:
            self.text_to_speech.say(str(score))
            self.text_to_speech.runAndWait()

    def commit_score(self, darts):
        if darts == 'q':
            exit(0)
        if type(darts) == str:
            darts = darts.split()
        
        points = 0
        for dart in darts:
            if (dart[0] not in ['S', 'T', 'D'] or dart[1:] not in [str(x) for x in range(1, 21)]) and dart not in ['SB', 'DB', 'miss']:
                print(f"Invalid dart: {dart}")
                return
            points += self.dartboard.get_score_for_dart(dart)
        
        self.num_visits_history[self.current_player][np.sum(self.leg_scores)] += 1
        
        self.scores[self.current_player] -= points
        
        if self.ruleset == 'x01':
            self.do_checks_x01_rules(darts, points)
        elif self.ruleset == '121':
            self.do_checks_121_rules(darts, points)
        

    def compute_remaining(self,darts,dartboard):
        """
        Inputs:
            darts (list[str]):
                List of dart results for the current visit.
                Empty strings represent missing or unused darts.
            dartboard (object):
                Dartboard object providing the method
                `get_score_for_dart(dart)` to compute scores.

        Returns:
            remaining (int or str):
                Remaining score after the throw, or "BUST"
                if the score becomes invalid (<= 1).

        Purpose:
            Calculates the remaining score for the current player
            based on the darts thrown during the visit. If the score
            goes below a valid finishing value, the turn is considered
            a bust.
        """

        # Compute remaining
        total = 0
        for d in darts:
            if d != '':
                total += dartboard.get_score_for_dart(d)

        remaining = self.scores[self.current_player] - total
        if remaining <= 1:
            remaining = "BUST"
        return remaining
    


    def do_checks_x01_rules(self, darts, points):
        num_visits = self.num_visits_history[self.current_player][np.sum(self.leg_scores)]
        
        if self.scores[self.current_player] == 0 and darts[-1][0] == 'D': # check out
            self.num_dart_history[self.current_player][np.sum(self.leg_scores)] += len(darts)
            self.averages[self.current_player] = ((self.averages[self.current_player] * num_visits-1) / num_visits) + ((points * 3/len(darts))/num_visits) 
            self.leg_scores[self.current_player] += 1
            self.scores = [self.x01] * self.num_players
            self.starting_player = (self.starting_player + 1) % self.num_players
            if max(self.leg_scores) == self.num_legs:
                if self.call_scores:
                    self.text_to_speech.say("Game shot, and the match.")
                    self.text_to_speech.runAndWait()
                exit(0)
            else:
                self.current_player = self.starting_player
                if self.call_scores:
                    self.text_to_speech.say(f"Game shot. {self.player_names[self.starting_player]} to throw in leg {sum(self.leg_scores)+1}.")
                    self.text_to_speech.runAndWait()

        else:
            if self.scores[self.current_player] <= 1: # bust
                self.scores[self.current_player] += points # Revert the points
                points = 0 # for average calculation

            self.num_dart_history[self.current_player][np.sum(self.leg_scores)] += 3 # any visit that isn't a checkout is 3 darts
            self.averages[self.current_player] = ((self.averages[self.current_player] * (num_visits-1)) / num_visits) + (points/num_visits)
            self.current_player = (self.current_player + 1) % self.num_players


    def do_checks_121_rules(self, darts, points):
        if self.scores[self.current_player] == 0 and darts[-1][0] == 'D': # check out
            self.leg_scores[self.current_player] += 1
            self.x01 += 1
            self.scores = [self.x01] * self.num_players
            self.starting_player = (self.starting_player + 1) % self.num_players

        elif self.scores[self.current_player] <= 1: # bust
            self.scores[self.current_player] += points # Revert the points
            self.current_player = (self.current_player + 1) % self.num_players

        else: # normal turn
            self.current_player = (self.current_player + 1) % self.num_players
        
        if len(self.point_history[self.current_player][np.sum(self.leg_scores)]) == 3: # this is 3rd visit
            self.scores = [121] * self.num_players