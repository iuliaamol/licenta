import os
import csv
import sys
import cv2
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from preprocessing.load_data2 import process_mammogram

def load_metrics(csv_path):
    valid_entries = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                iou = float(row['iou'])
                if iou > 0:
                    valid_entries.append(row['filename'])
            except:
                continue
    return valid_entries

def collect_image_paths(root_dir):
    image_map = {}
    for lesion_type in ["Benign", "Cancer"]:
        lesion_path = os.path.join(root_dir, lesion_type)
        for case in os.listdir(lesion_path):
            case_path = os.path.join(lesion_path, case)
            if not os.path.isdir(case_path):
                continue
            for fname in os.listdir(case_path):
                if fname.endswith("_CC.jpg") and "_Mask" not in fname:
                    image_map[fname] = os.path.join(case_path, fname)
    return image_map

def save_masks_for_valid_images(valid_filenames, image_map, sr_model_path, save_dir="saved_masks2", gt_dir="ground_truth_masks2",  labels_csv="mask_labels2.csv"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)

    with open(labels_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "label"])

        for filename in tqdm(valid_filenames, desc="Saving masks"):
            if filename not in image_map:
                print(f"Image path not found for: {filename}")
                continue

            image_path = image_map[filename]
            try:
                _, _, _, my_mask = process_mammogram(image_path, sr_model_path)

                # Salvează masca ca PNG
                mask_filename = filename.replace(".jpg", ".png")
                save_path = os.path.join(save_dir, mask_filename)
                cv2.imwrite(save_path, my_mask)


                #salveaza ground truth
                gt_mask_path = os.path.join(os.path.dirname(image_path), filename.replace(".jpg", "_Mask.jpg"))
                gt_mask_filename = filename.replace(".jpg", "_Mask.png")
                if os.path.exists(gt_mask_path):
                    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                    _, gt_binary = cv2.threshold(gt_mask, 50, 255, cv2.THRESH_BINARY)
                    gt_save_path = os.path.join(gt_dir, gt_mask_filename)
                    cv2.imwrite(gt_save_path, gt_binary)
                else:
                    print(f"Ground truth mask not found for {filename}")

                # Scrie în CSV eticheta (din numele folderului)
                label = "benign" if "Benign" in image_path else "cancer"
                writer.writerow([mask_filename, label])

            except Exception as e:
                print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    metrics_csv = "metrics_results2.csv"
    root_data_dir = "../data/MINI-DDSM-Complete-JPEG-8"
    sr_model_path = "../models/EDSR_x4.pb"

    valid_filenames = load_metrics(metrics_csv)
    image_map = collect_image_paths(root_data_dir)
    save_masks_for_valid_images(valid_filenames, image_map, sr_model_path)
