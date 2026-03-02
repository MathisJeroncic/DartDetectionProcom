import numpy as np

class GameLogic:
    def __init__(self, ruleset, player_names, x01=501, num_legs=1):
        self.ruleset = ruleset
        self.x01 = x01
        self.num_legs = num_legs
        self.player_names = player_names
        self.num_players = len(player_names)

        self.scores = [x01] * self.num_players
        self.leg_scores = [0] * self.num_players
        self.current_player = 0
        self.starting_player = 0

        self.num_visits_history = np.zeros((self.num_players, num_legs * 2 - 1))
        self.num_dart_history = np.zeros((self.num_players, num_legs * 2 - 1))
        self.averages = np.zeros(self.num_players)

        self.game_over = False
        self.winner = None
        self.last_event = None

    # ------------------ UTILS ------------------

    def get_score_for_dart(self, dart):
        if dart == 'DB':
            return 50
        if dart == 'SB':
            return 25
        if dart == 'miss':
            return 0

        number = int(dart[1:])
        if dart[0] == 'S':
            return number
        if dart[0] == 'D':
            return number * 2
        if dart[0] == 'T':
            return number * 3

    def validate_dart(self, dart):
        return (
            dart in ['SB', 'DB', 'miss'] or
            (dart[0] in ['S', 'D', 'T'] and dart[1:].isdigit() and 1 <= int(dart[1:]) <= 20)
        )

    def reset_game(self):
        """Remet la partie à son état initial pour recommencer."""
        self.scores = [self.x01] * self.num_players
        self.leg_scores = [0] * self.num_players
        self.current_player = self.starting_player
        self.num_dart_history = np.zeros((self.num_players, self.num_legs * 2 - 1))
        self.num_visits_history = np.zeros((self.num_players, self.num_legs * 2 - 1))
        self.averages = np.zeros(self.num_players)
        self.winner = None

    # ------------------ CORE ------------------

    def commit_score(self, darts):
        if self.game_over:
            return {"error": "Game already finished"}

        if isinstance(darts, str):
            darts = darts.split()

        for dart in darts:
            if not self.validate_dart(dart):
                return {"error": f"Invalid dart: {dart}"}

        points = sum(self.get_score_for_dart(d) for d in darts)
        player = self.current_player
        visit_index = int(np.sum(self.leg_scores))

        self.num_visits_history[player][visit_index] += 1
        self.scores[player] -= points

        event = {
            "player": self.player_names[player],
            "points": points,
            "remaining": self.scores[player],
            "checkout": False,
            "bust": False,
            "leg_won": False,
            "game_over": False,
            "next_player": None
        }

        if self.ruleset == "x01":
            self._check_x01(darts, points, event)
        elif self.ruleset == "121":
            self._check_121(darts, points, event)

        self.last_event = event
        return event

    # ------------------ RULES ------------------

    def _check_x01(self, darts, points, event):
        player = self.current_player
        visit_index = int(np.sum(self.leg_scores))

        # CHECKOUT
        if self.scores[player] == 0 and darts[-1][0] == 'D':
            self.num_dart_history[player][visit_index] += len(darts)
            self.leg_scores[player] += 1

            event["checkout"] = True
            event["leg_won"] = True

            if max(self.leg_scores) == self.num_legs:
                self.game_over = True
                self.winner = self.player_names[player]
                event["game_over"] = True
                return

            self.scores = [self.x01] * self.num_players
            self.starting_player = (self.starting_player + 1) % self.num_players
            self.current_player = self.starting_player
            event["next_player"] = self.player_names[self.current_player]
            return

        # BUST
        if self.scores[player] <= 1:
            self.scores[player] += points
            event["bust"] = True
            points = 0

        self.num_dart_history[player][visit_index] += 3
        self.current_player = (self.current_player + 1) % self.num_players
        event["remaining"] = self.scores[player]
        event["next_player"] = self.player_names[self.current_player]

    def _check_121(self, darts, points, event):
        player = self.current_player

        if self.scores[player] == 0 and darts[-1][0] == 'D':
            self.leg_scores[player] += 1
            self.x01 += 1
            self.scores = [self.x01] * self.num_players
            self.starting_player = (self.starting_player + 1) % self.num_players
            self.current_player = self.starting_player
            event["leg_won"] = True
        elif self.scores[player] <= 1:
            self.scores[player] += points
            event["bust"] = True
            self.current_player = (self.current_player + 1) % self.num_players
        else:
            self.current_player = (self.current_player + 1) % self.num_players

        event["next_player"] = self.player_names[self.current_player]