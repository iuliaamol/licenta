import cv2
from cv2 import dnn_superres
import matplotlib.pyplot as plt
import numpy as np


import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_mammogram(mammogram_path, sr_model_path, resize_dim=(256, 256), percentile_thresh=96):
    """
    Process a mammogram image to extract tumor contour and fitted ellipse.
    """
    # Load mammogram image in grayscale
    mammogram = cv2.imread(mammogram_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur and binary mask for breast
    blurred = cv2.GaussianBlur(mammogram, (5, 5), 0)
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Extract largest component: breast
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    breast_mask = np.zeros_like(binary_mask)
    breast_mask[labels == largest_label] = 255
    segmented = cv2.bitwise_and(blurred, blurred, mask=breast_mask)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(segmented)
    clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    sr_output_gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
    display_image = clahe_bgr

    # Thresholding and contour detection based on percentile
    threshold_value = np.percentile(sr_output_gray, percentile_thresh)
    _, tumor_mask = cv2.threshold(sr_output_gray, threshold_value, 255, cv2.THRESH_BINARY)

    # Apply distance transform and extract sure foreground
    dist = cv2.distanceTransform(tumor_mask, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # # Visualize sure_fg before applying contours
    # plt.figure(figsize=(6, 6))
    # plt.imshow(sure_fg, cmap='gray')
    # plt.title('Sure Foreground')
    # plt.axis('off')
    # plt.show()

    # Extract contours from the sure_fg (foreground) mask
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for selecting the brightest contour
    brightest_contour = None
    max_intensity = -1

    min_area = 2000
    max_area = 20000


    # Loop through each contour to find the brightest one based on intensity
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue  # Skip very small or very large contours

        # Create a mask from the contour
        mask = np.zeros_like(sr_output_gray)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        # Calculate mean intensity inside the contour
        mean_val = cv2.mean(sr_output_gray, mask=mask)[0]

        # Print contour details for debugging
        print(f"Contour area: {area}, Mean intensity: {mean_val}")

        # Update the brightest contour if this one has a higher mean intensity
        if mean_val > max_intensity:
            max_intensity = mean_val
            brightest_contour = cnt

    # Create a mask from the brightest contour
    mask_from_contour = np.zeros_like(sr_output_gray)
    if brightest_contour is not None and len(brightest_contour) >= 5:
        # Draw the brightest contour on the image
        cv2.drawContours(display_image, [brightest_contour], -1, (0, 255, 0), 2)

        # # Fit an ellipse to the brightest contour
        # ellipse = cv2.fitEllipse(brightest_contour)
        # cv2.ellipse(display_image, ellipse, (0, 0, 255), 2)

        # Create a binary mask for the brightest contour
        cv2.drawContours(mask_from_contour, [brightest_contour], -1, 255, -1)

    # # Visualize the final image with contours and ellipse
    # plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    # plt.title('Final Image with Tumor Contour and Ellipse')
    # plt.axis('off')
    # plt.show()

    return sr_output_gray, display_image, brightest_contour, mask_from_contour


# # Test the function with the image path
# mammogram_path = "../data/MINI-DDSM-Complete-JPEG-8/Benign/0033/C_0033_1.RIGHT_CC.jpg"
# sr_model_path = "../models/EDSR_x4.pb"
# sr_output_gray, display_image, brightest_contour, mask_from_contour = process_mammogram(mammogram_path, sr_model_path)
#
# # Visualize the final result with contours
# plt.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
# plt.title('Final Image with Tumor Contour and Ellipse')
# plt.axis('off')
# plt.show()




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

    # plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    # plt.title('mask')
    # plt.axis('off')
    # plt.show()



    return binary_mask


def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union (IoU) between two binary masks.
    """
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)

    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    tp = np.sum(intersection)
    fp = np.sum((mask1 == 0) & (mask2 == 1))
    fn = np.sum((mask1 == 1) & (mask2 == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou = tp / np.sum(union) if np.sum(union) > 0 else 0.0

    return iou, precision, recall


def visualize_mask_comparison(gt_mask, your_mask):
    """
    Overlay GT mask (blue) and your mask (green) for visual comparison.
    """

    overlay = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)

    overlay[gt_mask > 0] = [255, 0, 0]
    overlay[your_mask > 0] = [0, 255, 0]

    plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.title("Overlay: GT (Blue), Yours (Green)")
    plt.axis("off")
    plt.show()



mammogram_path = "../data/MINI-DDSM-Complete-JPEG-8/Benign/0033/C_0033_1.RIGHT_CC.jpg"
mask_path = "../data/MINI-DDSM-Complete-JPEG-8/Benign/0033/C_0033_1.RIGHT_CC_Mask.jpg"
sr_model_path = "../models/EDSR_x4.pb"

# mammogram_path = "../data/MINI-DDSM-Complete-JPEG-8/Cancer/1932/A_1932_1.LEFT_CC.jpg"
# mask_path = "../data/MINI-DDSM-Complete-JPEG-8/Cancer/1932/A_1932_1.LEFT_CC_Mask.jpg"
# sr_model_path = "../models/EDSR_x4.pb"
#

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



