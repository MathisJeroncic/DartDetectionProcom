import cv2
import numpy as np
import time
import threading
from get_scores import GetScores


class VideoProcessing:
    def __init__(self, model_dir="runs\\train108\\weights\\best.pt"):
        self.predict = GetScores(model_dir)
        self.dart_coords_in_visit = []
        self.wait_for_dart_removal = False
        self.pred_queue = -np.ones((5, 3, 2))
        self.pred_queue_count = 0
        self.repeat_threshold = 3
        self.scorer = None
        self.game_started = False
        self.manual_override = False

    # ---------- UTILS ----------
    def _distance(self, c1, c2):
        return np.linalg.norm(c1 - c2)

    def _adjust_coords(self, calibration_coords, dart_coords, resolution, crop_start, crop_size):
        calibration_coords = calibration_coords * resolution
        calibration_coords -= crop_start
        calibration_coords /= crop_size
        if dart_coords.shape != (0,):
            dart_coords = dart_coords * resolution
            dart_coords -= crop_start
            dart_coords /= crop_size
            dart_coords = dart_coords[np.all((dart_coords >= 0) & (dart_coords <= 1), axis=1)]
        return calibration_coords, dart_coords

    # ---------- PREDICTION ----------
    def _process_predictions(self, transformed_dart_coords):
        if transformed_dart_coords.shape == (0,):
            self.pred_queue[self.pred_queue_count % 5] = -np.ones((3, 2))
        else:
            filled = np.vstack((transformed_dart_coords, -np.ones((3 - len(transformed_dart_coords), 2))))
            self.pred_queue[self.pred_queue_count % 5] = filled

        self.pred_queue_count += 1
        if self.wait_for_dart_removal:
            count_empty = sum(np.all(frame == -1) for frame in self.pred_queue)
            if count_empty >= self.repeat_threshold:
                self._commit_score()
        else:
            valid = self.pred_queue[self.pred_queue != -1].reshape(-1, 2)
            if len(valid) == 0:
                return
            unique_preds = np.unique(valid, axis=0)
            matches = {tuple(up): [] for up in unique_preds}
            for frame in self.pred_queue:
                for pred in frame:
                    if np.any(pred == -1):
                        continue
                    for up in unique_preds:
                        if self._distance(pred, up) < 0.01:
                            matches[tuple(up)].append(pred)
                            break
            matches = {k: v for k, v in matches.items() if len(v) >= self.repeat_threshold}
            for v in matches.values():
                bp = np.mean(v, axis=0)
                if all(self._distance(bp, d) > 0.01 for d in self.dart_coords_in_visit):
                    if len(self.dart_coords_in_visit) < 3:
                        self.dart_coords_in_visit.append(bp)

    def _commit_score(self):
        if self.scorer:
            darts = [d for d in self.dart_coords_in_visit if d != ""]
            self.scorer.commit_score(darts)
        self.dart_coords_in_visit = []
        self.wait_for_dart_removal = False
        self.pred_queue = -np.ones((5, 3, 2))
        self.pred_queue_count = 0

    # ---------- STREAM ----------
    def start_stream(self, GUI, source, scorer, resolution=np.array([1200, 1600])):
        self.scorer = scorer
        self.game_started = True
        self.dart_coords_in_visit = []

        crop_size = min(resolution)
        crop_start = resolution / 2 - crop_size / 2

        cap = cv2.VideoCapture(f"http://{source}/video")
        if not cap.isOpened():
            print(f"❌ Impossible de se connecter à la caméra : {source}")
            self.game_started = False
            return

        prev_time = time.time()

        while self.game_started:
            ret, frame = cap.read()
            if not ret:
                continue

            # Ici tu pourras ajouter YOLO plus tard
            # calibration_coords, dart_coords = self.predict.process_yolo_output(frame)

            # Thread-safe update GUI
            GUI.root.after(0, GUI._display_frame, frame)

        cap.release()

    def start_stream_async(self, GUI, source, scorer, resolution=np.array([1200, 1600])):
        thread = threading.Thread(
            target=self.start_stream,
            args=(GUI, source, scorer, resolution),
            daemon=True
        )
        thread.start()

    # ---------- MANUAL OVERRIDE ----------
    def enter_manual_score(self, darts):
        self.manual_override = True
        self.dart_coords_in_visit = darts
        self._commit_score()
        self.manual_override = False