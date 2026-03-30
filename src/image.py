import cv2
import numpy as np

class Image():
    def __init__(self, img_path):
        """
        Inputs:
            img_path (str):
                Path to the image file to load.

        Returns:
            None

        Purpose:
            Loads an image from disk and initializes useful metadata
            such as image resolution and centered square crop parameters
            used for further image processing.
        """

        self.path = img_path
        self.img = self._load_image(img_path)

        self.resolution = np.array([
            self.img.shape[0],
            self.img.shape[1]
        ])

        self.crop_size = min(self.resolution)
        self.crop_start = self.resolution/2 - self.crop_size/2

    def _load_image(self, img_path):
        """
        Inputs:
            img_path (str):
                Path to the image file.

        Returns:
            img (np.ndarray):
                Image loaded as a NumPy array (BGR format from OpenCV).

        Purpose:
            Loads an image using OpenCV and validates that the file
            exists and is readable. Raises an explicit error if loading fails.
        """

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Impossible de charger l'image : {img_path}")
        return img