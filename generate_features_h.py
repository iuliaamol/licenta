import os
import pandas as pd
from tqdm import tqdm
from distance_features_h import extract_radial_features

def save_radial_features_to_csv(mask_folder, label_csv, out_csv_path, n_points=72):
    df = pd.read_csv(label_csv)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row["filename"]
        label = row["label"]
        mask_path = os.path.join(mask_folder, filename)

        de, dl, d = extract_radial_features(mask_path, n_points=n_points)
        if de is not None:
            entry = {"filename": filename, "lesion_type": label}
            for i in range(n_points):
                entry[f"de_{i}"] = de[i]
                entry[f"dl_{i}"] = dl[i]
                entry[f"d_{i}"] = d[i]
            results.append(entry)

    df_result = pd.DataFrame(results)
    df_result.to_csv(out_csv_path, index=False)
    print(f"Fișier salvat la: {out_csv_path}")


save_radial_features_to_csv(
    mask_folder="../evaluation/saved_masks2",
    label_csv="../evaluation/mask_labels2.csv",
    out_csv_path="../evaluation/distance_features_h.csv",
    n_points=72
)