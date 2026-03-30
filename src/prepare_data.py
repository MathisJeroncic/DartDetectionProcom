import os
import shutil
import random
from PIL import Image
import cv2
import numpy as np


import cv2
import numpy as np
import os


def prepare_extracted_image(frame,x,y,ratio_x,ratio_y):

    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Taille image in : ",frame.shape)
    
    center_coord=(x,y)
    cropped_frame=crop_around_point(frame, center=center_coord, ratio_x=ratio_x, ratio_y=ratio_y)
    cv2.imshow("Frame", cropped_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Taille image mid : ",cropped_frame.shape)


    prepared_frame=resize_images2(cropped_frame, size=(800, 800))
    print("Taille image out : ",prepared_frame.shape)
    cv2.imshow("Frame", prepared_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cropped_frame


def crop_around_point(
    img,
    center,      # (x, y) du centre réel
    ratio_x,        # taille du crop par rapport au min(h, w)
    ratio_y,
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
    if img is None:
        raise ValueError(f"Image non trouvée")

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

    return cropped




from PIL import Image
import numpy as np

def resize_images2(frame, size=(800, 800)):
    """
    Resize une image (numpy array) et retourne l'image redimensionnée.

    Parameters
    ----------
    frame : np.ndarray
        Image d'entrée (ex: frame OpenCV)

    size : tuple
        Nouvelle taille (width, height)

    Returns
    -------
    np.ndarray
        Image redimensionnée
    """

    if frame is None:
        raise ValueError("Image invalide")

    # Conversion numpy → PIL
    image = Image.fromarray(frame)

    # Resize
    image = image.resize(size, Image.BILINEAR)

    # Conversion PIL → numpy
    resized_frame = np.array(image)

    return resized_frame

def resize_images(frame, size=(800, 800)):
    os.makedirs(new_path, exist_ok=True)
    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    count = 0
    for image_name in os.listdir(path_to_images):
        if not image_name.lower().endswith(valid_ext):
            continue
        image_path = os.path.join(path_to_images, image_name)
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize(size, Image.BILINEAR)
            image.save(os.path.join(new_path, image_name))
            print(f"OK  → {image_name}")
            count += 1
        except Exception as e:
            print(f"ERREUR → {image_name} : {e}")

    print(f"\nTerminé : {count} images redimensionnées en {size[0]}×{size[1]}")


def change_bb_size_copy(src_label_dir, dst_label_dir, bb_size=0.025):
    os.makedirs(dst_label_dir, exist_ok=True)
    # Copier classes.txt tel quel s’il existe
    src_classes = os.path.join(src_label_dir, "classes.txt")
    dst_classes = os.path.join(dst_label_dir, "classes.txt")
    if os.path.exists(src_classes):
        shutil.copy(src_classes, dst_classes)
        print("OK → classes.txt (copié inchangé)")
    count_files = 0
    count_boxes = 0
    for file in os.listdir(src_label_dir):
        # ignorer classes.txt
        if not file.endswith(".txt") or file == "classes.txt":
            continue
        src_path = os.path.join(src_label_dir, file)
        dst_path = os.path.join(dst_label_dir, file)
        new_lines = []
        with open(src_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, _, _ = parts
                new_lines.append(
                    f"{cls} {x} {y} {bb_size:.6f} {bb_size:.6f}\n"
                )
                count_boxes += 1
        with open(dst_path, "w") as f:
            f.writelines(new_lines)
        count_files += 1
        print(f"OK → {file}")
    print(
        f"\nTerminé : {count_files} fichiers copiés, "
        f"{count_boxes} bounding boxes modifiées"
    )


def propagate_calibration_from_center(label_dir, output_dir, nb_calib, bb_size=0.025):
    """
    Propage les points de calibration (classes 0,1,2,3,5,6)
    à partir du centre (classe 7), en utilisant les distances
    moyennes à partir des nb_calibs premières images.

    - La classe 4 n'est jamais modifiée
    - classes.txt est copié tel quel
    """
    os.makedirs(output_dir, exist_ok=True)
    # ✅ Copier classes.txt tel quel
    src_classes = os.path.join(label_dir, "classes.txt")
    dst_classes = os.path.join(output_dir, "classes.txt")
    if os.path.exists(src_classes):
        shutil.copy(src_classes, dst_classes)
    CALIB_CLASSES = {0, 1, 2, 3, 5, 6}
    CENTER_CLASS = 7
    label_files = sorted([
        f for f in os.listdir(label_dir)
        if f.endswith(".txt") and f != "classes.txt"
    ])
    calib_files = label_files[:nb_calib]
    offsets = {cls: [] for cls in CALIB_CLASSES}

    # Calcul des offsets moyens
    for file in calib_files:
        center = None
        points = {}
        with open(os.path.join(label_dir, file), "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, _, _ = parts
                cls = int(cls)
                x, y = float(x), float(y)
                if cls == CENTER_CLASS:
                    center = (x, y)
                elif cls in CALIB_CLASSES:
                    points[cls] = (x, y)
        if center is None:
            continue
        cx, cy = center
        for cls, (x, y) in points.items():
            offsets[cls].append((x - cx, y - cy))
    mean_offsets = {
        cls: (
            sum(v[0] for v in vals) / len(vals),
            sum(v[1] for v in vals) / len(vals)
        )
        for cls, vals in offsets.items()
        if len(vals) > 0
    }

    # Application aux fichiers
    for file in label_files:
        path_in = os.path.join(label_dir, file)
        path_out = os.path.join(output_dir, file)
        lines = []
        present_classes = set()
        center = None
        with open(path_in, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = parts
                cls_i = int(cls)
                present_classes.add(cls_i)
                if cls_i == CENTER_CLASS:
                    center = (float(x), float(y))
                lines.append(line)

        if center is None:
            with open(path_out, "w") as f:
                f.writelines(lines)
            continue
        cx, cy = center
        for cls in CALIB_CLASSES:
            if cls not in present_classes and cls in mean_offsets:
                dx, dy = mean_offsets[cls]
                lines.append(
                    f"{cls} {cx + dx:.6f} {cy + dy:.6f} {bb_size:.6f} {bb_size:.6f}\n"
                )
        with open(path_out, "w") as f:
            f.writelines(lines)

        print(f"OK → {file}")
    print("\nPropagation terminée – classes.txt copié")


def remap_classes(label_dir, output_dir):
    """
    Transforme les noms/classes dans les fichiers .txt selon le mapping :

    15 → 0
    16 → 1
    17 → 2
    18 → 3
    21 → 4
    19 → 5
    20 → 6

    label_dir : dossier contenant les labels originaux
    output_dir : dossier où seront écrits les labels modifiés
    """
    os.makedirs(output_dir, exist_ok=True)
    # Copier classes.txt tel quel
    src_classes = os.path.join(label_dir, "classes.txt")
    dst_classes = os.path.join(output_dir, "classes.txt")
    if os.path.exists(src_classes):
        shutil.copy(src_classes, dst_classes)
        print("OK → classes.txt (copié inchangé)")
    # Mapping des classes
    class_map = {
        15: "0",
        16: "1",
        17: "2",
        18: "3",
        21: "4",
        19: "5",
        20: "6"
    }
    # Parcourir tous les fichiers .txt (sauf classes.txt)
    for file_name in sorted(os.listdir(label_dir)):
        if not file_name.endswith(".txt") or file_name == "classes.txt":
            continue
        path_in = os.path.join(label_dir, file_name)
        path_out = os.path.join(output_dir, file_name)
        new_lines = []
        with open(path_in, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, w, h = parts
                cls = int(cls)
                cls_new = class_map.get(cls, str(cls))
                new_lines.append(f"{cls_new} {x} {y} {w} {h}\n")
        with open(path_out, "w") as f:
            f.writelines(new_lines)

        print(f"OK → {file_name}")
    print(f"\nTerminé : remapping terminé (classes.txt préservé)")
        

def split_dataset(dataset_name, val_frac=0.1, test_frac=0.15):
    path_to_data = "data\\darts"
    path_to_labels = os.path.join(path_to_data, 'labels', dataset_name)
    path_to_images = path_to_labels.replace('labels', 'images')

    image_names = os.listdir(path_to_images)
    random.shuffle(image_names)

    num_val = int(len(image_names)*val_frac)
    num_test = int(len(image_names)*test_frac)

    for ds_type in ['train', 'val', 'test']:
        os.makedirs(os.path.join(path_to_labels, ds_type))
        os.makedirs(os.path.join(path_to_images, ds_type))

    for image_name in image_names[:num_val]:
        shutil.move(os.path.join(path_to_images, image_name), os.path.join(path_to_images, 'val'))
        shutil.move(os.path.join(path_to_labels, image_name.replace(image_name[-4:], '.txt')), os.path.join(path_to_labels, 'val'))

    for image_name in image_names[num_val:num_val+num_test]:
        shutil.move(os.path.join(path_to_images, image_name), os.path.join(path_to_images, 'test'))
        shutil.move(os.path.join(path_to_labels, image_name.replace(image_name[-4:], '.txt')), os.path.join(path_to_labels, 'test'))
    
    for image_name in image_names[num_val+num_test:]:
        shutil.move(os.path.join(path_to_images, image_name), os.path.join(path_to_images, 'train'))
        shutil.move(os.path.join(path_to_labels, image_name.replace(image_name[-4:], '.txt')), os.path.join(path_to_labels, 'train'))

