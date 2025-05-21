"axa orizontala"

import os
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_mask_comparison_with_ellipse(my_mask_path, gt_mask_path, save_path, padding=20):
    # Încarcă măștile
    my_mask = cv2.imread(my_mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)

    if my_mask is None or gt_mask is None:
        print("Eroare la citirea fișierelor.")
        return

    # Binarizare
    my_mask = (my_mask > 0).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    # Contururi
    contours_my, _ = cv2.findContours(my_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_gt, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Imagine RGB pentru desen
    overlay = np.zeros((my_mask.shape[0], my_mask.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(overlay, contours_gt, -1, (255, 0, 0), 2)   # Roșu = GT
    cv2.drawContours(overlay, contours_my, -1, (0, 255, 0), 2)   # Verde = Masca detectată

    # Dacă putem trasa elipsa
    if contours_my and len(contours_my[0]) >= 5:
        # Fit elipsă inițial
        ellipse = cv2.fitEllipse(contours_my[0])
        (cx, cy), (MA, ma), angle = ellipse

        # Forțăm elipsa să aibă axa majoră pe orizontală (adică unghi = 0)
        # MA = major axis, ma = minor axis, le păstrăm ca valori
        if MA < ma:  # Ne asigurăm că MA este major axis
            MA, ma = ma, MA

        ellipse_horizontal = ((cx, cy), (MA, ma), 0.0)  # unghiul = 0 → orizontal

        # Desenează elipsa rotită (pe orizontală) – galben
        cv2.ellipse(overlay, ellipse_horizontal, (0, 255, 255), 2)

    # Bounding box pentru decupare
    all_points = np.vstack([cnt for cnt in (contours_my + contours_gt) if cnt is not None])
    x, y, w, h = cv2.boundingRect(all_points)
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, overlay.shape[1])
    y2 = min(y + h + padding, overlay.shape[0])

    # Decupare și salvare imagine
    cropped = overlay[y1:y2, x1:x2]
    cv2.imwrite(save_path, cropped)


    # # Afișare
    # plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    # plt.title(" GT (blue), My mask (green), Ellipse (yellow)")
    # plt.axis("off")
    # plt.tight_layout()
    # plt.show()



def save_all(saved_mask_dir, gt_mask_dir, output_dir="masks+ellipse_visuals", limit=None):
    os.makedirs(output_dir, exist_ok=True)
    mask_files = sorted([f for f in os.listdir(saved_mask_dir) if f.endswith(".png")])

    if limit:
        mask_files = mask_files[:limit]

    for fname in tqdm(mask_files, desc="Salvez imagini"):
        my_mask_path = os.path.join(saved_mask_dir, fname)
        gt_mask_path = os.path.join(gt_mask_dir, fname.replace(".png", "_Mask.png"))
        save_filename = fname.replace(".png", "_fig1.png")
        save_path = os.path.join(output_dir, save_filename)

        if os.path.exists(gt_mask_path):
            plot_mask_comparison_with_ellipse(my_mask_path, gt_mask_path,save_path)
        else:
            print(f" Ground truth lipsește pentru {fname}")



save_all("../evaluation/saved_masks2", "../evaluation/ground_truth_masks2", output_dir="masks+ellipse_visuals_h", limit=None)


import cv2
import numpy as np

def extract_radial_features(mask_path, n_points=72):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Eroare la citire: {mask_path}")
        return None, None, None

    # Binarizare
    mask = (mask > 0).astype(np.uint8)

    # Găsire contur
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours or len(contours[0]) < 5:
        return None, None, None

    contour = contours[0]

    # Fit elipsă
    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (MA, ma), _ = ellipse

    # Forțăm orientare orizontală
    if MA < ma:
        MA, ma = ma, MA  # MA = major axis, ma = minor axis

    a = MA / 2  # semiaxa mare
    b = ma / 2  # semiaxa mică
    center = np.array([cx, cy])
    angle_rad = 0.0  # elipsa este pe orizontală
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])

    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    de = []
    dl = []

    def find_contour_intersection(contour, center, direction, max_angle_cos=0.98):
        for pt in contour:
            vec = pt[0] - center
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            vec_unit = vec / norm
            if np.dot(vec_unit, direction) > max_angle_cos:
                return norm
        return np.nan  # dacă nu se găsește nicio intersecție

    for theta in angles:
        # Punct pe elipsă în direcția theta
        pt_ellipse = center + R @ np.array([a * np.cos(theta), b * np.sin(theta)])
        vec = pt_ellipse - center
        vec /= (np.linalg.norm(vec) + 1e-8)  # direcție unitară

        de.append(np.linalg.norm(pt_ellipse - center))
        dl_val = find_contour_intersection(contour, center, vec)
        dl.append(dl_val)

    # Conversie în numpy arrays
    de = np.array(de)
    dl = np.array(dl)
    d = np.abs(de - dl)

    return de, dl, d
