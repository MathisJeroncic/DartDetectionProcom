from ultralytics import YOLO
import numpy as np

# Initialize a YOLO model using the provided weights file
def initYOLO(weights):
    """
    Inputs:
        weights (str): Path to the YOLO weights file.

    Returns:
        model (YOLO): Initialized YOLO model ready for inference.

    Purpose:
        Loads and initializes a YOLO model using the specified weights.
        This function centralizes model creation for easier reuse.
    """
    model = YOLO(weights)
    return model


# Run YOLO prediction on a single image
def predictYOLO(image, model):
    """
    Inputs:
        image (object): Image container expected to have an attribute `img`
                        containing the image array.
        model (YOLO): Initialized YOLO model.

    Returns:
        result (ultralytics.engine.results.Results):
            YOLO prediction result for a single image.

    Purpose:
        Runs object detection on one image and retrieves the first
        prediction result (since only one image is processed).
    """
    result = model.predict(image.img, verbose=False)[0]  # 0 because only one image
    return result


# Extract calibration points and dart coordinates from YOLO output
def extract_darts_cal_coords_from_yolo_output(yoloOutput):
    """
    Inputs:
        yoloOutput (Results):
            YOLO prediction output containing detected boxes,
            classes, and confidence scores.

    Returns:
        tuple:
            calibration_coords (np.ndarray shape (6,2)):
                Normalized coordinates of calibration points.
            dart_coords (np.ndarray shape (N,2)):
                Normalized coordinates of detected darts (max 3).

        OR

        dict:
            {"error": str} if calibration cannot be performed.

    Purpose:
        Parses YOLO detection results to extract:
        - Calibration marker coordinates (high-confidence detections)
        - Dart impact positions

        The function filters detections, prevents overwriting calibration
        points, limits dart detections to three, and validates whether
        enough calibration points are available.
    """

    # Initialize calibration coordinates (6 points, x/y)
    # -1 indicates missing calibration detections
    calibration_coords = -np.ones((6, 2))

    # List to store detected dart coordinates
    dart_coords = []

    # Extract YOLO detection data
    classes = yoloOutput.boxes.cls
    boxes = yoloOutput.boxes.xywhn
    conf = yoloOutput.boxes.conf

    # Iterate over all detections
    for i in range(len(classes)):

        # Convert class tensor to integer
        cls = int(classes[i].item())

        # Safely extract normalized coordinates from tensors
        x = boxes[i][0].detach().cpu().item()
        y = boxes[i][1].detach().cpu().item()

        # ---- DARTS ----
        # Class 4 corresponds to darts
        if cls == 4:
            # Keep at most 3 darts
            if len(dart_coords) < 3:
                dart_coords.append([x, y])
            continue

        # ---- CALIBRATION ----
        # Ignore low-confidence detections
        if conf[i].item() < 0.85:
            continue

        calibration_i = cls

        # Skip dart class index and shift calibration indices
        if calibration_i > 4:
            calibration_i -= 1

        # Avoid overwriting existing calibration points
        if np.all(calibration_coords[calibration_i] == -1):
            calibration_coords[calibration_i] = [x, y]

    # Convert dart list to numpy array
    dart_coords = np.array(dart_coords) if len(dart_coords) > 0 else np.empty((0, 2))

    # Check if too many calibration points are missing
    if np.count_nonzero(calibration_coords == -1) / 2 > 2:
        return {"error": "Calibration impossible sur cette image"}

    return calibration_coords, dart_coords