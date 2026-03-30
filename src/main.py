import model
import image
import dartboard
import game_logic
import video_processing
import sys
import numpy as np

# Path to input image containing the dartboard
img_path = "../data/pictures/d1_02_04_2020 IMG_1093.JPG"

# Path to trained YOLO weights
modelWeights = "../weights/weights.pt"






def predictOnImage():
    # ------------------------------------------------------------
    # 1. LOAD IMAGE
    # ------------------------------------------------------------
    # Create Image object (loads image and prepares metadata)
    myImage = image.Image(img_path)
    
    # ------------------------------------------------------------
    # 2. LOAD YOLO MODEL AND RUN DETECTION
    # ------------------------------------------------------------
    # Initialize YOLO model from weights
    modelYOLO = model.initYOLO(modelWeights)

    # Run object detection on the image
    yoloPrediction = model.predictYOLO(myImage.img, modelYOLO)

    # Extract calibration points and detected dart coordinates
    calibration_coords, dart_coords = model.extract_darts_cal_coords_from_yolo_output(
        yoloPrediction
    )

    # ------------------------------------------------------------
    # 3. DARTBOARD GEOMETRY & HOMOGRAPHY
    # ------------------------------------------------------------
    # Create dartboard model containing geometry and scoring logic
    myDartboard = dartboard.Dartboard()

    # Compute homography matrix aligning image space
    # with normalized dartboard reference plane
    H_matrix = myDartboard.find_homography_matrix(
        calibration_coords,
        myImage.crop_size
    )

    # ------------------------------------------------------------
    # 4. TRANSFORM DART COORDINATES
    # ------------------------------------------------------------
    # Map dart coordinates from camera/image perspective
    # into canonical dartboard coordinates
    transformed_dart_coords = myDartboard.apply_homography(
        H_matrix[0],
        dart_coords,
        myImage.crop_size
    )

    # ------------------------------------------------------------
    # 5. INITIALIZE GAME LOGIC
    # ------------------------------------------------------------
    # Create a game instance (x01 ruleset)
    game = game_logic.GameLogic(
        myDartboard,
        ruleset='x01',
        player_names=['Kamon'],
        x01=1001,
        num_legs=1
    )

    # ------------------------------------------------------------
    # 6. COMPUTE SCORE
    # ------------------------------------------------------------
    # Determine dart labels (T20, D5, etc.) and total score
    darts, score = myDartboard.score(np.array(transformed_dart_coords))
    
    # Compute remaining score for the current player
    remaining = game.compute_remaining(darts, myDartboard)
    
    # ------------------------------------------------------------
    # 7. FORMAT RESULTS
    # ------------------------------------------------------------
    # Store relevant outputs in a dictionary
    results = {
        # "darts_coords": transformed_dart_coords,
        "dart_labels": darts,
        "score": score,
        # "remaining": remaining,
        # "calibration": calibration_coords,
        # "raw_dart_coords": dart_coords,
        # "homography": H_matrix
    }
    
    # ------------------------------------------------------------
    # 8. DISPLAY RESULTS
    # ------------------------------------------------------------
    # Print scoring results
    print(results)

    # Print only the image filename
    print(img_path.split("/")[-1])

def predictOnVideo():
    source='../data/videos/video_test_auto_crop.mp4'

    modelYOLO = model.initYOLO(modelWeights)
    myDartboard = dartboard.Dartboard()

    videoProc=video_processing.VideoProcessing()

    game = game_logic.GameLogic(ddartboard=myDartboard,
        ruleset='x01',
        player_names=['Kamon'],
        x01=1001,
        num_legs=1
    )

    videoProc.start(modelYOLO,myDartboard,source,game,np.array((1200, 1600)))
    

if __name__ == "__main__":
    # Vérifie qu'un argument a été fourni
    if len(sys.argv) < 2:
        print("Usage : uv run main.py [image|video]")
        sys.exit(1)

    argument = sys.argv[1].lower()

    if argument == "image":
        predictOnImage()
    elif argument == "video":
        predictOnVideo()
    else:
        print("Argument invalide. Utilise 'image' ou 'video'.")
