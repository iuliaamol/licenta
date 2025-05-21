import pandas as pd
import numpy as np

# Încarcă fișierul CSV cu vectorii d
csv_path = "../evaluation/distance_features.csv"
df = pd.read_csv(csv_path)

# Extrage doar coloanele d_0...d_71
d_cols = [col for col in df.columns if col.startswith("d_")]

# Grupează pe tipul leziunii
for lesion_type in ["benign", "cancer"]:
    print(f"\n Statistici pentru: {lesion_type.upper()}")

    subset = df[df["lesion_type"] == lesion_type][d_cols]

    # Calculează statistici pe întreg vectorul d
    all_d_values = subset.values.flatten()
    all_d_values = all_d_values[~np.isnan(all_d_values)]  # elimină NaN

    print(f"Număr total de valori d: {len(all_d_values)}")
    print(f"Min:   {np.min(all_d_values):.4f}")
    print(f"Max:   {np.max(all_d_values):.4f}")
    print(f"Mean:  {np.mean(all_d_values):.4f}")
    print(f"Std:   {np.std(all_d_values):.4f}")
