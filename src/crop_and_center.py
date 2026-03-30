import cv2
import numpy as np
import os

def crop_dartboard(
    image_path,
    output_path,
    margin_ratio=0.15,
    debug=False
):
    """
    Détecte la cible de fléchettes et crop l'image autour.

    margin_ratio : marge autour du rayon détecté (15% recommandé)
    debug        : affiche les étapes intermédiaires
    """

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image non trouvée: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.5)

    # Détection de cercles (Hough)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=200,
        param1=100,
        param2=30,
        minRadius=100,
        maxRadius=0
    )

    if circles is None:
        raise RuntimeError("Aucune cible détectée")

    # On prend le plus grand cercle détecté (logique pour une cible)
    circles = np.uint16(np.around(circles))
    circles = sorted(circles[0], key=lambda c: c[2], reverse=True)
    x, y, r = circles[0]

    # Ajout marge
    r_crop = int(r * (1 + margin_ratio))

    h, w, _ = img.shape
    x1 = max(x - r_crop, 0)
    y1 = max(y - r_crop, 0)
    x2 = min(x + r_crop, w)
    y2 = min(y + r_crop, h)

    cropped = img[y1:y2, x1:x2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cropped)

    if debug:
        debug_img = img.copy()
        cv2.circle(debug_img, (x, y), r, (0, 255, 0), 3)
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow("Detection", debug_img)
        cv2.imshow("Cropped", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cropped.shape


def crop_center(image_path, output_path, ratio=0.8):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image non trouvée: {image_path}")
    h, w, _ = img.shape
    size = int(min(h, w) * ratio)
    cx, cy = w // 2, h // 2
    half = size // 2
    cropped =  img[cy-half:cy+half, cx-half:cx+half] 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cropped)


def crop_around_point(
    image_path,
    output_path,
    center=None,      # (x, y) du centre réel
    ratio_x=0.8,        # taille du crop par rapport au min(h, w)
    ratio_y=0.8,
    auto_clip=True    # évite de sortir des bords
):
    """
    Crop carré centré autour d'un point donné et sauvegarde l'image.

    Parameters
    ----------
    image_path : str
        Chemin de l'image d'entrée.

    output_path : str
        Chemin complet du fichier de sortie (DOIT inclure le nom + extension).

    center : tuple(int, int), optional
        Coordonnées (cx, cy) du centre réel de la cible.
        Si None → centre géométrique de l'image.

    ratio : float, optional (default=0.8)
        Taille du crop en proportion du plus petit côté de l'image.
        Doit être compris entre 0 et 1.

    Returns
    -------
    cropped : np.ndarray
        Image rognée (utile si tu veux l'utiliser dans un pipeline).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image non trouvée: {image_path}")

    h, w = img.shape[:2]

    # Taille du carré
    size_x = int(w * ratio_x)
    size_y = int(h * ratio_y)
    half_x = size_x // 2
    half_y = size_y // 2

    # Si aucun centre fourni → centre géométrique
    if center is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = center

    # Coordonnées initiales
    x1 = cx - half_x
    x2 = cx + half_x
    y1 = cy - half_y
    y2 = cy + half_y

    if auto_clip:
        # Ajustement si on dépasse les bords
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > w:
            x1 -= (x2 - w)
            x2 = w
        if y2 > h:
            y1 -= (y2 - h)
            y2 = h

        # Sécurité finale
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

    cropped = img[y1:y2, x1:x2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cropped)

    print(f"Saved: {output_path}")



def get_img_size(img_path):
    img = cv2.imread(img_path)
    print(img.shape)

def process_dataset(input_dir, output_dir, center_coords, ratio_x = 0.8, ratio_y = 0.8):
    for file in os.listdir(input_dir):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            in_path = os.path.join(input_dir, file)
            out_path = os.path.join(output_dir, file)

            try:
                #crop_center(in_path, out_path, ratio = 0.83)
                crop_around_point(in_path, out_path, center_coords, ratio_x, ratio_y)
                print(f"[OK] {file}")
            except Exception as e:
                print(f"[FAIL] {file} → {e}")




if __name__ == "__main__":
    in_path = "./data/darts/images/fine_tuning_matinee_raw/"
    out_path = "./data/darts/images/fine_tuning_matinee_cropped/"
    center_coord = (1500, 2590) # A determiner par tatonement en fonction de la configuration des images
    ratio_x = 0.6
    ratio_y = 0.58
    #get_img_size(in_path)
    #crop_around_point(in_path, out_path, center=center_coord, ratio_x=ratio_x, ratio_y=ratio_y)
    process_dataset(in_path, out_path, center_coord, ratio_x, ratio_y)
