from PIL.ImageOps import scale

from ultralytics import YOLO
import dartboard
import model
from auto_crop import ParamCropper

import numpy as np
import time
import cv2


class VideoProcessing:
    """
    Classe principale responsable du traitement vidéo :
    - détection des fléchettes via YOLO
    - calibration du plateau
    - suivi des fléchettes au fil des frames
    - calcul et validation des scores
    """

    def __init__(self, model_dir="weights.pt"):
        """
        Initialise le système de traitement vidéo.

        Args:
            model_dir (str): chemin vers le modèle YOLO entraîné.
        """
        # Coordonnées des fléchettes détectées pendant une visite (max 3)
        self.dart_coords_in_visit = []

        # Outil de crop automatique centré sur le dartboard
        self.cropper = ParamCropper(
            ratio_w=0.7, ratio_h=0.7, center_x=220, center_y=560, output_size=(800, 800)
        )

    def _distance(self, coord1, coord2):
        """
        Calcule la distance euclidienne entre deux coordonnées 2D.

        Args:
            coord1 (np.array): point 1
            coord2 (np.array): point 2

        Returns:
            float: distance entre les deux points
        """
        return np.sqrt(np.sum((coord1 - coord2) ** 2))

    def _assess_visit(self, darts,ddartboard):
        """
        Évalue une visite (jusqu'à 3 fléchettes).

        - Calcule le score total
        - Détermine le score restant
        - Gère les cas de BUST
        - Déclenche la validation du score si nécessaire

        Args:
            darts (list): liste des fléchettes détectées

        Returns:
            tuple: (score_visite, score_restant)
        """
        darts = [dart for dart in darts if dart != '']

        score = 0
        for dart in darts:
            score += ddartboard.get_score_for_dart(dart)

        remaining = self.scorer.scores[self.scorer.current_player] - score

        # Validation automatique si 3 fléchettes ou fin de tour
        if remaining <= 1 or len(darts) == 3:
            if self.wait_for_dart_removal == False:
                self.scorer.read_score(score)

            self.wait_for_dart_removal = True
        else:
            self.wait_for_dart_removal = False

        # Gestion du bust (règles x01)
        if (remaining == 0 and darts[-1][0] != 'D') or remaining == 1 or remaining < 0:
            remaining = 'BUST'

        return score, remaining

    def _commit_score(self):
        """
        Valide définitivement le score de la visite.

        Réinitialise :
        - coordonnées des fléchettes
        - calibration utilisateur
        - file de prédictions
        """
        self.scorer.commit_score(
            [dart for dart in self.darts_in_visit if dart != '']
        )

        self.dart_coords_in_visit, self.darts_in_visit = [], [''] * 3
        self.user_calibration = -np.ones((6, 2))
        self.wait_for_dart_removal = False
        self.pred_queue = -np.ones((5, 3, 2))
        self.pred_queue_count = 0

    def _adjust_coords(self, calibration_coords, dart_coords,
                       resolution, crop_start, crop_size):
        """
        Ajuste les coordonnées après application d’un crop carré.

        Convertit :
        coordonnées normalisées -> pixels -> crop -> normalisées.

        Args:
            calibration_coords: points de calibration
            dart_coords: positions des fléchettes
            resolution: résolution image originale
            crop_start: origine du crop
            crop_size: taille du crop

        Returns:
            tuple: coordonnées ajustées
        """

        calibration_coords *= resolution
        calibration_coords -= crop_start
        calibration_coords /= crop_size

        if len(dart_coords) > 0:
            dart_coords *= resolution
            dart_coords -= crop_start
            dart_coords /= crop_size

            # Supprime les points hors du crop
            dart_coords = dart_coords[
                np.all(np.logical_and(dart_coords >= 0,
                                      dart_coords <= 1), axis=1)
            ]

        return calibration_coords, dart_coords

    def _process_predictions(self, transformed_dart_coords,
                             repeat_threshold):
        """
        Stabilise les détections sur plusieurs frames.

        Objectifs :
        - filtrer le bruit YOLO
        - confirmer une fléchette seulement si répétée
        - détecter le retrait des fléchettes

        Args:
            transformed_dart_coords: coords projetées sur le plateau
            repeat_threshold (int): nb minimum de répétitions
        """

        # Ajout des prédictions dans une FIFO queue (5 frames)
        if len(transformed_dart_coords) == 0:
            self.pred_queue[self.pred_queue_count % 5] = -np.ones((3, 2))
        else:
            self.pred_queue[self.pred_queue_count % 5] = np.vstack(
                (
                    transformed_dart_coords,
                    -np.ones((3 - len(transformed_dart_coords), 2))
                )
            )

        self.pred_queue_count += 1

        # Détection du retrait des fléchettes
        if self.wait_for_dart_removal:
            count = 0
            for frame in self.pred_queue:
                if np.all(frame == -1):
                    count += 1

            if count >= repeat_threshold:
                self._commit_score()

        # Ajout de nouvelles fléchettes
        elif self.darts_in_visit.count('') > 0:

            valid_preds = self.pred_queue[self.pred_queue != -1].reshape(-1, 2)
            valid_preds = np.round(valid_preds, decimals=3)

            unique_predictions = np.unique(valid_preds, axis=0)

            matches = {tuple(pred): [] for pred in unique_predictions}

            # Regroupe les prédictions similaires
            for frame in self.pred_queue:
                for pred in frame:
                    if np.any(pred == -1):
                        continue
                    for unique_pred in unique_predictions:
                        if self._distance(pred, unique_pred) < 0.04:
                            matches[tuple(unique_pred)].append(pred)
                            break

            # Garde seulement les prédictions stables
            matches = {
                k: v for k, v in sorted(
                    matches.items(),
                    key=lambda item: len(item[1]),
                    reverse=True
                )
                if len(v) >= repeat_threshold
            }

            best_predictions = [
                np.mean(match_, axis=0)
                for match_ in matches.values()
            ]

            # Ajout si nouvelle fléchette
            for best_pred in best_predictions:

                if len(self.dart_coords_in_visit) >= 3:
                    break

                is_new_dart = all(
                    self._distance(coords, best_pred) > 0.04
                    for coords in self.dart_coords_in_visit
                )

                if is_new_dart:
                    self.dart_coords_in_visit.append(best_pred)

    def start(self, modelYOLO,ddartboard,source, scorer, resolution: np.array):
        """
        Lance la boucle principale de traitement vidéo.

        Pipeline :
        1. Lecture frame vidéo
        2. Crop automatique
        3. Détection YOLO
        4. Calibration plateau
        5. Projection homographique
        6. Stabilisation des détections
        7. Calcul score
        8. Affichage GUI

        Args:
            gui: interface graphique
            source: source vidéo
            scorer: moteur de score du jeu
            resolution: résolution caméra
        """

        self.scorer = scorer
        self.num_corrections = 0

        crop_size = min(resolution)

        self.dart_coords_in_visit, self.darts_in_visit = [], [''] * 3
        self.user_calibration = -np.ones((6, 2))
        self.wait_for_dart_removal = False
        self.game_over = False

        # File FIFO des prédictions
        self.pred_queue = -np.ones((5, 3, 2))
        self.pred_queue_count = 0
        repeat_threshold = 3

        prev_frame_time = 0
        new_frame_time = 0

        self.frames_since_three_darts = 0

        cap = cv2.VideoCapture(source)

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            # Crop automatique du plateau
            frame_cropped, crop_x, crop_y, crop_size = \
                self.cropper.crop_frame(frame)

            # Inférence YOLO
            yoloPrediction = model.predictYOLO(frame_cropped, modelYOLO)
            #results = model(frame_cropped)
            #result = results[0]

            #print("Boxes detected:", len(result.boxes))
            #print(result.boxes.cls)

            frame = frame_cropped

            # Affichage debug
            display = cv2.resize(frame_cropped, None, fx=0.7, fy=0.7)
            cv2.imshow("Cropped", display)
            cv2.waitKey(1)

            if self.game_over:
                break

            # Extraction calibration + fléchettes
            calibration_coords, dart_coords = \
                model.extract_darts_cal_coords_from_yolo_output(yoloPrediction)

            print("Calibration:", calibration_coords)
            print("Darts:", dart_coords)

            if np.count_nonzero(calibration_coords == -1) / 2 > 2:
                continue

            calibration_coords = np.where(
                self.user_calibration == -1,
                calibration_coords,
                self.user_calibration
            )

            # Calcul homographie plateau
            H_matrix = ddartboard.find_homography_matrix(
                calibration_coords, crop_size
            )

            # Conversion coordonnées fléchettes
            if len(dart_coords) == 0:
                dart_coords_np = np.empty((0, 2))
            else:
                dart_coords_np = np.array(
                    dart_coords, dtype=np.float32
                )

            transformed_dart_coords = \
                ddartboard.apply_homography(
                    H_matrix[0], dart_coords_np, crop_size#Le [0] pas sur
                )

            # Stabilisation
            self._process_predictions(
                transformed_dart_coords,
                repeat_threshold
            )

            # Scoring
            self.darts_in_visit, score = \
                ddartboard.score(
                    np.array(self.dart_coords_in_visit)
                )

            while len(self.darts_in_visit) < 3:
                self.darts_in_visit.append('')

            score, remaining = self._assess_visit(self.darts_in_visit,ddartboard)

            # Auto validation si 3 fléchettes visibles longtemps
            if self.darts_in_visit.count('') == 0:
                self.frames_since_three_darts += 1
            else:
                self.frames_since_three_darts = 0

            if self.frames_since_three_darts > 20:
                self._commit_score()
                self.frames_since_three_darts = 0

            print(
                f"Current visit darts: {self.darts_in_visit}, "
                f"Score for visit: {score}, Remaining score: {remaining}"
            )

            # Calcul FPS
            new_frame_time = time.time()
            fps = round(1 / (new_frame_time - prev_frame_time), 1)
            prev_frame_time = new_frame_time


        print("H_matrix shape:", H_matrix[0].shape)
        print(f'Number of user corrections: {self.num_corrections}')
        print(f'Number of darts thrown: '
              f'{np.sum(self.scorer.num_dart_history)}')