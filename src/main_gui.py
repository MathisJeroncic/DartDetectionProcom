import tkinter as tk
from game_logic import GameLogic
from gui_video import GUIVideo

def main_gui(video = True):
    if video :
        app = GUIVideo()
        app.launch()

if __name__ == "__main__":
    main_gui()