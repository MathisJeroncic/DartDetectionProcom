import cv2
import numpy as np
import torch

class Dartboard():
    def __init__(self):  # de base ya un attribut model_dir="runs\\train108\\weights\\best.pt"
        """
        Inputs:
            None

        Returns:
            None

        Purpose:
            Initializes dartboard geometry, scoring regions, segment layout,
            and reference calibration coordinates used to map detected dart
            positions onto a normalized dartboard plane.
        """

        #self.model_dir = model_dir

        self.class_names = {0: '20', 1: '3', 2: '11', 3: '6', 4: 'dart'}
        
        # dart board measurements in mm
        ring = 10.0  # width of the double and treble rings
        bullseye_wire = 1.6  # width of the bullseye wires
        wire = 1.0  # width of other wires

        self.scoring_names = np.array(['DB', 'SB', 'S', 'T', 'S', 'D', 'miss'])
        self.scoring_radii = np.array([0, 6.35, 15.9, 107.4-ring, 107.4, 170.0-ring, 170.0])
        self.scoring_radii[1:3] += (bullseye_wire/2)

        # normalize radii between 0 and 1 using board diameter
        self.scoring_radii /= 451.0

        self.segment_angles = np.array([-9, 9, 27, 45, 63, -81, -63, -45, -27])
        self.segment_numbers = np.array(([6,11], [10,14], [15,9], [2,12],
                                         [17,5], [19,1], [7,18], [16,4], [8,13]))

        # computing the boardplane calibration coordinates
        self.boardplane_calibration_coords = -np.ones((6, 2))
        h = self.scoring_radii[-1]

        # for 20 & 3
        a = h*np.cos(np.deg2rad(81))
        o = (h**2 - a**2)**0.5
        self.boardplane_calibration_coords[0] = [0.5 - a, 0.5 - o]
        self.boardplane_calibration_coords[1] = [0.5 + a, 0.5 + o]

        # for 11 & 6
        a = h*np.cos(np.deg2rad(-9))
        o = (h**2 - a**2)**0.5
        self.boardplane_calibration_coords[2] = [0.5 - a, 0.5 + o]
        self.boardplane_calibration_coords[3] = [0.5 + a, 0.5 - o]

        # for 9 & 15
        a = h*np.cos(np.deg2rad(27))
        o = (h**2 - a**2)**0.5
        self.boardplane_calibration_coords[4] = [0.5 - a, 0.5 - o]
        self.boardplane_calibration_coords[5] = [0.5 + a, 0.5 + o]
    

    def find_homography_matrix(self, calibration_coords, image_shape):
        """
        Inputs:
            calibration_coords (np.ndarray shape (6,2)):
                Normalized calibration points detected in the image.
            image_shape (array-like):
                Image dimensions used to convert normalized coordinates
                into pixel coordinates.

        Returns:
            H_matrix (tuple):
                Homography matrix computed by OpenCV mapping image space
                to the normalized dartboard plane.

        Purpose:
            Computes the homography transformation aligning detected
            calibration points with the ideal dartboard reference plane.
        """

        mask = np.all(
            np.logical_and(calibration_coords >= 0, calibration_coords <= 1),
            axis=1
        )
        H_matrix = cv2.findHomography(
            calibration_coords[mask]*image_shape,
            self.boardplane_calibration_coords[mask]*image_shape
        )
        return H_matrix
    

    def apply_homography(self, matrix, dart_coords, image_shape):
        """
        Inputs:
            matrix (np.ndarray):
                Homography transformation matrix.
            dart_coords (np.ndarray shape (N,2)):
                Normalized dart coordinates detected in the image.
            image_shape (array-like):
                Image dimensions used for coordinate conversion.

        Returns:
            transformed_darts (np.ndarray shape (N,2)):
                Dart coordinates mapped into normalized dartboard space.

        Purpose:
            Applies the homography transformation to detected dart
            coordinates so they can be evaluated in the canonical
            dartboard coordinate system for scoring.
        """

        if len(dart_coords) == 0:
            return dart_coords
       
        homogenous_coords = np.concatenate(
            (dart_coords*image_shape, np.ones((dart_coords.shape[0], 1))),
            axis=1
        ).T

        transformed_darts = matrix @ homogenous_coords
        transformed_darts /= transformed_darts[-1]
        transformed_darts = transformed_darts[:-1].T
        transformed_darts /= image_shape

        return transformed_darts
    

    def score(self, transformed_darts):
        """
        Inputs:
            transformed_darts (np.ndarray shape (N,2)):
                Dart coordinates expressed in normalized dartboard space.

        Returns:
            darts (list[str]):
                List describing each dart result (e.g., 'T20', 'D5', 'SB').
            score (int):
                Total score obtained from all darts.

        Purpose:
            Determines the scoring region and segment number for each dart
            based on angular position and distance from the board center,
            then computes individual and total scores.
        """

        darts = ['' for _ in range(len(transformed_darts))]
        score = 0

        if len(darts) == 0:
            return darts, score

        mask = transformed_darts[:,0] == 0.5
        transformed_darts[mask,0] += 0.00001
        
        # compute dart angles relative to center
        angles = np.rad2deg(
            np.arctan((transformed_darts[:,1]-0.5) /
                      (transformed_darts[:,0]-0.5))
        )
        angles = np.where(angles > 0, np.floor(angles), np.ceil(angles))

        for i in range(len(transformed_darts)):
            dart_coords = transformed_darts[i]
            
            if abs(angles[i]) >= 81:
                possible_numbers = np.array([3,20])
            else:
                possible_numbers = self.segment_numbers[
                    np.where(
                        self.segment_angles ==
                        max(self.segment_angles[self.segment_angles <= angles[i]])
                    )
                ][0]

            if all(possible_numbers == [6,11]):
                coord_index = 0
            else:
                coord_index = 1

            if dart_coords[coord_index] > 0.5:
                number = possible_numbers[0]
            else:
                number = possible_numbers[1]
            
            distance = ((dart_coords[0]-0.5)**2 +
                        (dart_coords[1]-0.5)**2)**0.5

            region = self.scoring_names[
                np.argmax(self.scoring_radii[distance > self.scoring_radii])
            ]

            scores = {
                'DB':['DB',50],
                'SB':['SB',25],
                'S':['S'+str(number), number],
                'T':['T'+str(number), number*3],
                'D':['D'+str(number), number*2],
                'miss':['miss',0]
            }
            
            darts[i] = scores[region][0]
            score += scores[region][1]

            while len(darts) < 3:
                darts.append('')
                
        return darts, score
    

    def get_score_for_dart(self, dart):
        """
        Inputs:
            dart (str):
                Dart notation (e.g., 'T20', 'D5', 'SB', 'DB', 'miss').

        Returns:
            int:
                Numerical score associated with the dart.

        Purpose:
            Converts a dart notation into its corresponding numerical
            score according to standard dart rules.
        """

        if dart == 'DB':
            return 50
        elif dart == 'SB':
            return 25
        elif dart == 'miss':
            return 0
        else:
            number = int(dart[1:])
            if dart[0] == 'S':
                return number
            elif dart[0] == 'D':
                return number*2
            elif dart[0] == 'T':
                return number*3