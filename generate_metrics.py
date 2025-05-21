import os
import csv

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from  preprocessing.load_data import process_mammogram, process_ground_truth_mask, calculate_iou



def collect_images_with_mask(root_dir):
    """"
     look for all images paired with a mask
     -goes through each folder, and each file
     -looks for .jpg (without the masks)
     -generates its mask file name
     -checks it the image has a mask
     return a list of tuples : image_path, mask_path, lesion_type
     """
    image_pairs = []

    for lesion_type in ["Benign", "Cancer"]:
        lesion_path = os.path.join(root_dir, lesion_type) #..data/MINI-DDSM-Complete-JPEG-8/Benign
        for case in os.listdir(lesion_path):
            case_folder = os.path.join(lesion_path, case)
            if not os.path.isdir(case_folder):
                continue
            for fname in os.listdir(case_folder): # goes through each file
                if fname.endswith("_CC.jpg") and "_Mask" not in fname:
                    image_path = os.path.join(case_folder, fname)
                    mask_path = image_path.replace(".jpg", "_Mask.jpg")
                    if os.path.exists(mask_path):
                        image_pairs.append((image_path, mask_path, lesion_type.lower()))
    return image_pairs



def extract_density(filename):
    first_letter=os.path.basename(filename)[0].upper()
    if first_letter in ["A","B","C","D"]:
        return first_letter
    return "unknown"



def process_all_images(image_pairs, sr_model_path, output_csv="metrics_results2.csv"):
    """
    take the image_pairs and return a csv with results
    """
    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "lesion_type", "density", "iou", "precision", "recall"])

        for image_path, mask_path, lesion_type in tqdm(image_pairs, desc="Processing"):
            filename = os.path.basename(image_path)
            density = extract_density(filename)

            try:
                # loads the image and its mask
                sr_image, result_img, contour, my_mask = process_mammogram(image_path, sr_model_path)


                gt_mask = process_ground_truth_mask(mask_path)
                if gt_mask.shape != my_mask.shape:
                    print(
                        f"Shape mismatch: {os.path.basename(image_path)}: GT {gt_mask.shape} vs Pred {my_mask.shape}")
                    gt_mask = cv2.resize(gt_mask, (my_mask.shape[1], my_mask.shape[0]), interpolation=cv2.INTER_NEAREST)


                # computes metrics
                iou, precision, recall = calculate_iou(gt_mask, my_mask)

                # writes in file
                if iou > 0:
                    writer.writerow([
                        filename,
                        lesion_type,
                        density,
                        round(iou, 4),
                        round(precision, 4),
                        round(recall, 4)
                    ])

            except Exception as e:
                print(f"error  {filename}: {e}")




if __name__ == "__main__":
    root_data_dir = "../data/MINI-DDSM-Complete-JPEG-8"
    sr_model = "../models/EDSR_x4.pb"

    image_pairs = collect_images_with_mask(root_data_dir)
    print(f" Found {len(image_pairs)} images with mask.")

    process_all_images(image_pairs, sr_model, output_csv="metrics_results.csv")
    print("done!")
