import cv2
from cv2 import dnn_superres
import matplotlib.pyplot as plt
import numpy as np


import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_mammogram(mammogram_path, sr_model_path=None, resize_dim=(256, 256), percentile_thresh=99):
    """
    Process a mammogram image to extract tumor contour and fitted ellipse.

    Returns:
        sr_output_gray: grayscale image
        display_image: image with contour + ellipse
        brightest_contour: detected contour
        mask_from_contour: binary mask created from brightest contour
    """
    # Load mammogram
    mammogram = cv2.imread(mammogram_path, cv2.IMREAD_GRAYSCALE)

    # Gaussian blur and mask the breast
    blurred = cv2.GaussianBlur(mammogram, (5, 5), 0)
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    breast_mask = np.zeros_like(binary_mask)
    breast_mask[labels == largest_label] = 255
    segmented = cv2.bitwise_and(blurred, blurred, mask=breast_mask)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(segmented)
    clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    sr_output_gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
    display_image = clahe_bgr.copy()

    # Thresholding
    threshold_value = np.percentile(sr_output_gray, percentile_thresh)
    _, tumor_mask = cv2.threshold(sr_output_gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Opening + Closing
    kernel = np.ones((3, 3), np.uint8)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Optional: mask out image borders
    valid_mask = np.zeros_like(tumor_mask)
    valid_mask[30:-30, 30:-30] = 255  # ignore 30px left/right
    tumor_mask = cv2.bitwise_and(tumor_mask, valid_mask)

    # Contour detection
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    brightest_contour = None
    max_intensity = -1

    min_area = 3000
    max_area = 20000

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        img_h, img_w = sr_output_gray.shape
        if x < 30 or (x + w) > (img_w - 30):
            continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-5)
        if circularity < 0.2:
            continue

        mask = np.zeros_like(sr_output_gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val = cv2.mean(sr_output_gray, mask=mask)[0]

        if mean_val > max_intensity:
            max_intensity = mean_val
            brightest_contour = cnt

    mask_from_contour = np.zeros_like(sr_output_gray)
    if brightest_contour is not None and len(brightest_contour) >= 5:
        cv2.drawContours(display_image, [brightest_contour], -1, (0, 255, 0), 2)
        ellipse = cv2.fitEllipse(brightest_contour)
        cv2.ellipse(display_image, ellipse, (0, 0, 255), 2)
        cv2.drawContours(mask_from_contour, [brightest_contour], -1, 255, -1)

    # # Afișare (opțională)
    # plt.imshow(sr_output_gray, cmap='gray')
    # plt.title('Super-Resolved Grayscale Image')
    # plt.axis('off')
    # plt.show()
    #
    # plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    # plt.title('Image with Contour and Ellipse')
    # plt.axis('off')
    # plt.show()
    #
    # plt.imshow(mask_from_contour, cmap='gray')
    # plt.title('Mask from Brightest Contour')
    # plt.axis('off')
    # plt.show()

    return sr_output_gray, display_image, brightest_contour, mask_from_contour



#processing the ground truth mask
def process_ground_truth_mask(mask_path, threshold_value=50):
    """
    Load and resize a ground truth mask using direct stretching (no aspect ratio preservation).
    Returns a binary mask (0/255) aligned with the predicted mask.
    """
    # Load grayscale mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found at path: {mask_path}")

    # Binarize
    _, binary_mask = cv2.threshold(mask, threshold_value, 255, cv2.THRESH_BINARY)
    #
    # plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    # plt.title('mask')
    # plt.axis('off')
    # plt.show()
    #


    return binary_mask


def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    Automatically fills contours if masks contain only outlines.
    """
    # Resize if needed
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Asigură-te că sunt binare
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    # Umple contururile (dacă sunt doar linii)
    def fill_mask(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, -1)
        return (filled > 0).astype(np.uint8)

    filled1 = fill_mask(mask1)
    filled2 = fill_mask(mask2)

    # Calcul metrice
    intersection = np.logical_and(filled1, filled2)
    union = np.logical_or(filled1, filled2)

    tp = np.sum(intersection)
    fp = np.sum((filled1 == 0) & (filled2 == 1))
    fn = np.sum((filled1 == 1) & (filled2 == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou = tp / np.sum(union) if np.sum(union) > 0 else 0.0

    return iou, precision, recall





mammogram_path = "../data/MINI-DDSM-Complete-JPEG-8/Benign/0033/C_0033_1.RIGHT_CC.jpg"
mask_path = "../data/MINI-DDSM-Complete-JPEG-8/Benign/0033/C_0033_1.RIGHT_CC_Mask.jpg"
sr_model_path = "../models/EDSR_x4.pb"

# mammogram_path = "../data/MINI-DDSM-Complete-JPEG-8/Cancer/3065/B_3065_1.RIGHT_CC.jpg"
# mask_path = "../data/MINI-DDSM-Complete-JPEG-8/Cancer/3065/B_3065_1.RIGHT_CC_Mask.jpg"
# sr_model_path = "../models/EDSR_x4.pb"


#my defined mask from mammography
sr_image, result_img, contour, my_mask = process_mammogram(mammogram_path, sr_model_path)




#ground truth mask (deformed)
gt_mask=process_ground_truth_mask(mask_path)


# Convert the SR image to BGR for drawing
overlay = cv2.cvtColor(sr_image, cv2.COLOR_GRAY2BGR)

#contour from my mask
my_contours, _ = cv2.findContours(my_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay, my_contours, -1, (0, 255, 0), 2)


#contour from the ground truth mask
gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay, gt_contours, -1, (255, 0, 0), 2)

plt.figure()
plt.imshow(overlay, cmap='gray')
plt.title(f"masks")
plt.axis("off")
plt.show()

# Evaluate (my mask and the ground truth mask)
iou, precision, recall = calculate_iou(gt_mask, my_mask)

print("Evaluation Metrics:")
print(f"IoU:       {iou:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")