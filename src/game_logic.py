import numpy as np
import pyttsx3

class GameLogic:
    def __init__(self, ruleset, player_names = ['Player 1', 'Player 2'], x01=501, num_legs=1, call_scores=True):
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