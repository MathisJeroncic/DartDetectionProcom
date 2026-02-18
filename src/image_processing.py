from ultralytics import YOLO
from get_scores import GetScores
from game_logic import GameLogic
import numpy as np
import cv2
import time

class ImageProcessing:
    def __init__(self, model_dir="../weights/weights.pt"):
        print("image_processing.py: __init__\n")
        self.model = YOLO(model_dir)
        self.predict = GetScores(model_dir)

    def _adjust_coords(self, calibration_coords, dart_coords, resolution, crop_start, crop_size):
        print("image_processing.py: _adjust_coords\n")
        calibration_coords *= resolution   # normalised -> pixel
        calibration_coords -= crop_start
        calibration_coords /= crop_size

        if dart_coords.shape != (0,):
            dart_coords *= resolution
            dart_coords -= crop_start
            dart_coords /= crop_size
            dart_coords = dart_coords[
                np.all((dart_coords >= 0) & (dart_coords <= 1), axis=1)
            ]

        return calibration_coords, dart_coords

    def process(self, image_path, scorer):
        print("image_processing.py: process\n")
        """Process a single image and return dart scores."""

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")

        # Resolution of input
        resolution = np.array([img.shape[0], img.shape[1]])

        crop_size = min(resolution)
        crop_start = resolution/2 - crop_size/2

        # Run YOLO once
        result = self.model.predict(img, verbose=False)[0]

        # Extract predictions
        calibration_coords, dart_coords = self.predict.process_yolo_output(result)
        if np.count_nonzero(calibration_coords == -1)/2 > 2:
            return {"error": "Calibration impossible sur cette image",
                    "filepath": image_path}

        # Adjust for square crop
        calibration_coords, dart_coords = self._adjust_coords(
            calibration_coords,
            dart_coords,
            resolution,
            crop_start,
            crop_size
        )

        # Compute homography
        H_matrix = self.predict.find_homography(calibration_coords, crop_size)

        # Transform darts to board plane
        transformed_dart_coords = self.predict.transform_to_boardplane(
            H_matrix[0],
            dart_coords,
            crop_size
        )

        # Compute scoring
        darts, score = self.predict.score(np.array(transformed_dart_coords))
        while len(darts) < 3:
            darts.append('')

        # Compute remaining
        total = 0
        for d in darts:
            if d != '':
                total += scorer.get_score_for_dart(d)

        remaining = scorer.scores[scorer.current_player] - total
        if remaining <= 1:
            remaining = "BUST"

        return {
            # "darts_coords": transformed_dart_coords,
            "dart_labels": darts,
            "score": score,
            # "remaining": remaining,
            # "calibration": calibration_coords,
            # "raw_dart_coords": dart_coords,
            # "homography": H_matrix
        }


if __name__ == "__main__":
    print("image_processing.py: __main__\n")
    game = GameLogic(ruleset='x01', player_names=['Kamon'], x01=1001, num_legs=1)

    img_path = "../data/pictures/d1_02_04_2020 IMG_1093.JPG"
    image_processing = ImageProcessing()
    results = image_processing.process(img_path, game)
    print(results)
    print(img_path.split("/")[-1])
    #pass
