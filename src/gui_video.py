import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

from video_processing import VideoProcessing
from game_logic import GameLogic


class GUIVideo:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Darts – Video Prototype")

        self.video = VideoProcessing()
        self.game = None
        self.video_running = False

    def launch(self):
        self.start_screen()
        self.root.mainloop()

    # ---------- START SCREEN ----------
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

        ttk.Label(frame, text="Camera IP (ex: 192.168.1.42)").pack()
        self.camera_ip_entry = ttk.Entry(frame, width=30)
        self.camera_ip_entry.insert(0, "192.168.1.42")
        self.camera_ip_entry.pack(pady=5)

        ttk.Button(frame, text="Start Game", command=self.init_game).pack(pady=10)

    def init_game(self):
        players = [p.strip() for p in self.players_entry.get().split(",")]
        x01 = int(self.score_entry.get())
        self.camera_ip = self.camera_ip_entry.get().strip()
        self.game = GameLogic("x01", players, x01=x01)
        self.game_screen()

    # ---------- GAME SCREEN ----------
    def game_screen(self):
        self.clear()

        main = ttk.Frame(self.root, padding=10)
        main.pack()

        # --- Canvas vidéo ---
        self.canvas = tk.Canvas(main, width=640, height=640, bg="black")
        self.canvas.grid(row=0, column=0, padx=10)

        # --- Panneau droit ---
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

        ttk.Button(self.side_frame, text="Start Video", command=self.start_video).pack(fill="x")
        ttk.Button(self.side_frame, text="Enter score manually", command=self.manual_score).pack(fill="x", pady=5)
        ttk.Button(self.side_frame, text="Restart", command=self.start_screen).pack(fill="x")

        self.update_display()

    # ---------- VIDEO ----------
    def start_video(self):
        if self.video_running:
            return
        self.video_running = True
        self.video.start_stream_async(
            GUI=self,
            source=self.camera_ip,
            scorer=self.game
        )
        self.pending_label.config(text="Video started…")

    def _display_frame(self, frame, score=None, remaining=None, fps=None):
        """Thread-safe update du canvas via Tkinter after()"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (640, 640))
        img = Image.fromarray(frame_rgb)
        self.tk_img = ImageTk.PhotoImage(img)

        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        if score is not None:
            text = f"Score détecté: {score}"
            if remaining is not None:
                text += f" | Remaining: {remaining}"
            if fps is not None:
                text += f" | FPS: {fps}"
            self.pending_label.config(text=text)

    # ---------- MANUAL SCORE ----------
    def manual_score(self):
        self.pending_label.config(text="Enter darts manually:")
        self.manual_entry = ttk.Entry(self.side_frame, width=20)
        self.manual_entry.pack(pady=5)
        self.manual_submit = ttk.Button(self.side_frame, text="Submit", command=self.submit_manual)
        self.manual_submit.pack(pady=5)

    def submit_manual(self):
        darts = self.manual_entry.get().split()
        self.game.commit_score(darts)
        self.update_display()
        self.pending_label.config(text=f"Manual input: {darts}")
        self.manual_entry.destroy()
        self.manual_submit.destroy()

    # ---------- UI ----------
    def update_display(self):
        for i, p in enumerate(self.game.player_names):
            txt = f"{p}: {self.game.scores[i]}"
            if i == self.game.current_player:
                txt += "  ←"
            self.score_labels[i].config(text=txt)
        self.turn_label.config(text=f"Current player: {self.game.player_names[self.game.current_player]}")

    # ---------- CLEAR ----------
    def clear(self):
        for w in self.root.winfo_children():
            w.destroy()


if __name__ == "__main__":
    GUIVideo().launch()