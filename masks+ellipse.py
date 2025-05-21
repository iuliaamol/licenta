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

    # Imagine RGB
    overlay = np.zeros((my_mask.shape[0], my_mask.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(overlay, contours_gt, -1, (255, 0, 0), 2)   # Roșu
    cv2.drawContours(overlay, contours_my, -1, (0, 255, 0), 2)   # Verde

    # Elipsă (dacă e posibil)
    if contours_my and len(contours_my[0]) >= 5:
        ellipse = cv2.fitEllipse(contours_my[0])
        cv2.ellipse(overlay, ellipse, (0, 255, 255), 2)  # Galben

    # Bounding box total
    all_points = np.vstack([cnt for cnt in (contours_my + contours_gt) if cnt is not None])
    x, y, w, h = cv2.boundingRect(all_points)
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, overlay.shape[1])
    y2 = min(y + h + padding, overlay.shape[0])

    # Decupare
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



save_all("../evaluation/saved_masks2", "../evaluation/ground_truth_masks2", output_dir="masks+ellipse_visuals", limit=None)
