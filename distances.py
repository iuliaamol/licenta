def plot_full_radial_figure(my_mask_path, n_points=72, angle_deg_highlight=0, save_path=None):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Load and preprocess mask
    mask = cv2.imread(my_mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Eroare la citire: {my_mask_path}")
        return

    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours or len(contours[0]) < 5:
        print("Contur invalid.")
        return

    contour = contours[0]
    ellipse = cv2.fitEllipse(contour)
    center = np.array(ellipse[0])
    a, b = ellipse[1][0] / 2, ellipse[1][1] / 2
    angle_rad = np.deg2rad(ellipse[2])
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])

    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    ellipse_distances = []
    contour_distances = []
    directions = []

    def find_intersection(contour, direction):
        max_dot = 0.98
        for pt in contour:
            vec = pt[0] - center
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            vec_norm = vec / norm
            if np.dot(vec_norm, direction) > max_dot:
                return pt[0]
        return None

    for theta in angles:
        pt_ellipse = center + R @ np.array([a * np.cos(theta), b * np.sin(theta)])
        dir_vec = pt_ellipse - center
        dir_vec /= (np.linalg.norm(dir_vec) + 1e-8)
        pt_contour = find_intersection(contour, dir_vec)

        ellipse_distances.append(np.linalg.norm(pt_ellipse - center))
        if pt_contour is not None:
            contour_distances.append(np.linalg.norm(pt_contour - center))
        else:
            contour_distances.append(np.nan)
        directions.append((pt_ellipse, pt_contour))

    # ------------ FIGURE ------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Subfigura 1 – Imagine
    overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)       # green: contour
    cv2.ellipse(overlay, ellipse, (0, 255, 255), 2)                # yellow: ellipse
    cv2.circle(overlay, tuple(np.round(center).astype(int)), 4, (255, 105, 180), -1)  # pink: center

    idx = int(angle_deg_highlight / 360 * n_points)
    pt_ellipse, pt_contour = directions[idx]

    if pt_ellipse is not None:
        cv2.circle(overlay, tuple(np.round(pt_ellipse).astype(int)), 4, (0, 255, 255), -1)  # yellow dot
        cv2.line(overlay,
                 tuple(np.round(center).astype(int)),
                 tuple(np.round(pt_ellipse).astype(int)),
                 (0, 255, 255), 2)

    if pt_contour is not None:
        cv2.circle(overlay, tuple(np.round(pt_contour).astype(int)), 4, (0, 0, 255), -1)    # red dot
        cv2.line(overlay,
                 tuple(np.round(center).astype(int)),
                 tuple(np.round(pt_contour).astype(int)),
                 (255, 0, 0), 2)

    # Cropping
    all_pts = np.vstack((contour.reshape(-1, 2), [pt_ellipse], [pt_contour], [center]))
    x, y, w, h = cv2.boundingRect(all_pts.astype(np.int32))
    pad = 30
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, overlay.shape[1])
    y2 = min(y + h + pad, overlay.shape[0])
    cropped = overlay[y1:y2, x1:x2]

    axs[0].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f"Ellipse and Contour  {angle_deg_highlight}°")
    axs[0].axis("off")

    # Subfigura 2 – Grafic
    ellipse_distances = np.array(ellipse_distances)
    contour_distances = np.array(contour_distances)
    diff = np.abs(ellipse_distances - contour_distances)

    axs[1].plot(ellipse_distances, label="Ellipse", color="blue")
    axs[1].plot(contour_distances, label="Contour", color="green")
    axs[1].plot(diff, label="|Difference|", color="red")
    axs[1].set_title(" Distance contours")
    axs[1].set_xlabel("Angle Index")
    axs[1].set_ylabel("Distance to Center (pixels)")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


plot_full_radial_figure("../evaluation/saved_masks2/C_0033_1.RIGHT_CC.png", angle_deg_highlight=0)
