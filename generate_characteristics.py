import os

import pandas as pd
import numpy as np

# Citim fișierul
df = pd.read_csv("../evaluation/distance_features.csv")
# === Pas 2: Extragem vectorii d_0 ... d_71 ===
d_columns = [f'd_{i}' for i in range(72)]
d_vectors = df[d_columns].values

# === Pas 3: Calculăm cele 3 caracteristici pentru fiecare rând ===
theta = 2 * np.pi / 72  # pasul unghiular

Rq_list = []
Rdq_list = []
Circularity_list = []

for d in d_vectors:
    # Rq: root mean square roughness
    Rq = np.sqrt(np.mean(d ** 2))

    # Rdq: root mean square slope
    slope = (d[1:] - d[:-1]) / theta
    Rdq = np.sqrt(np.mean(slope ** 2))

    # Circularity
    circularity = (np.mean(d) ** 2) / np.mean(d ** 2)

    Rq_list.append(Rq)
    Rdq_list.append(Rdq)
    Circularity_list.append(circularity)
import pandas as pd

# Setări pentru afișare completă în consolă
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Adăugăm caracteristicile
df['Rq'] = Rq_list
df['Rdq'] = Rdq_list
df['Circularity'] = Circularity_list

output_folder = "../features"
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, "characteristics_full.csv")

df[['filename', 'lesion_type', 'Rq', 'Rdq', 'Circularity']].to_csv(output_path, index=False)
print(f"✅ Caracteristicile au fost salvate în: {output_path}")