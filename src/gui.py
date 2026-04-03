import tkinter as tk
from tkinter import ttk
from game_logic import GameLogic


class GUIVideo:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Darts – Video Prototype")
        self.game = None

    def launch(self):
        self.start_screen()
        self.root.mainloop()

    def start_screen(self):
        self.clear()

        frame = ttk.Frame(self.root, padding=20)
        frame.pack()

        ttk.Label(frame, text="Players (comma separated):").pack()
        self.players_entry = ttk.Entry(frame, width=30)
        self.players_entry.insert(0, "Rémi, Alexis")
        self.players_entry.pack(pady=5)

        ttk.Label(frame, text="Starting score (x01):").pack()
        self.score_entry = ttk.Entry(frame, width=10)
        self.score_entry.insert(0, "301")
        self.score_entry.pack(pady=5)

        ttk.Button(frame, text="Start Game", command=self.init_game).pack(pady=10)

    def init_game(self):
        players = [p.strip() for p in self.players_entry.get().split(",")]
        x01 = int(self.score_entry.get())
        self.game = GameLogic("x01", players, x01=x01)
        self.game_screen()

    def game_screen(self):
        self.clear()

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(main, width=640, height=640, bg="black")
        self.canvas.grid(row=0, column=0, padx=10)

        self.side_frame = ttk.Frame(main)
        self.side_frame.grid(row=0, column=1, sticky="n")

        ttk.Label(self.side_frame, text="Scores", font=("Arial", 14, "bold")).pack(anchor="w")

        self.score_labels = {}
        for i, p in enumerate(self.game.player_names):
            lbl = ttk.Label(self.side_frame)
            lbl.pack(anchor="w")
            self.score_labels[i] = lbl

        self.turn_label = ttk.Label(self.side_frame)
        self.turn_label.pack(pady=5)

        self.pending_label = ttk.Label(self.side_frame, foreground="blue")
        self.pending_label.pack(pady=5)

        ttk.Separator(self.side_frame).pack(fill="x", pady=10)

        ttk.Button(self.side_frame, text="Enter score manually", command=self.manual_score).pack(fill="x", pady=5)
        ttk.Button(self.side_frame, text="Restart", command=self.start_screen).pack(fill="x")

        # ---------- STATS (now under scores) ----------
        ttk.Separator(self.side_frame).pack(fill="x", pady=10)

        ttk.Label(self.side_frame, text="Stats", font=("Arial", 12, "bold")).pack(anchor="w")

        self.stats_labels = {}
        for i, p in enumerate(self.game.player_names):
            lbl = ttk.Label(self.side_frame)
            lbl.pack(anchor="w", pady=2)
            self.stats_labels[i] = lbl

        self.update_display()

    def manual_score(self):
        self.pending_label.config(text="Enter darts manually:")
        self.manual_entry = ttk.Entry(self.side_frame, width=20)
        self.manual_entry.pack(pady=5)

        self.manual_submit = ttk.Button(
            self.side_frame,
            text="Submit",
            command=self.submit_manual
        )
        self.manual_submit.pack(pady=5)

    def submit_manual(self):
        darts = self.manual_entry.get().split()
        self.game.commit_score(darts)
        self.update_display()

        self.pending_label.config(text=f"Manual input: {darts}")

        self.manual_entry.destroy()
        self.manual_submit.destroy()

    def update_display(self):
        for i, p in enumerate(self.game.player_names):
            txt = f"{p}: {self.game.scores[i]}"
            if i == self.game.current_player:
                txt += "  ←"
            self.score_labels[i].config(text=txt)

            stats = (
                f"{p} | Avg/visit: {self.game.averages[i]:.2f} | "
                f"D: {self.game.num_doubles[i]} | "
                f"T: {self.game.num_triples[i]}"
            )
            self.stats_labels[i].config(text=stats)

        self.turn_label.config(
            text=f"Current player: {self.game.player_names[self.game.current_player]}"
        )

    def clear(self):
        for w in self.root.winfo_children():
            w.destroy()


if __name__ == "__main__":
    GUIVideo().launch()