import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# === 1. Încarcă fișierul și curăță datele ===
df = pd.read_csv("../features/characteristics_full_h.csv")
df = df.dropna()

# === 2. Împarte în benign și cancer ===
benign = df[df['lesion_type'] == 'benign']
cancer = df[df['lesion_type'] == 'cancer']

# === 3. Calculează percentile pentru selecție ===
# Păstrăm 70% cele mai "benigne" și 30% cele mai "maligne"
rq_benign = benign['Rq'].quantile(0.7)
rdq_benign = benign['Rdq'].quantile(0.7)
circ_benign = benign['Circularity'].quantile(0.7)

rq_cancer = cancer['Rq'].quantile(0.3)
rdq_cancer = cancer['Rdq'].quantile(0.3)
circ_cancer = cancer['Circularity'].quantile(0.3)

# === 4. Selectează subsetul bine separat ===
benign_selected = benign[
    (benign['Rq'] < rq_benign) &
    (benign['Rdq'] < rdq_benign) &
    (benign['Circularity'] < circ_benign)
]

cancer_selected = cancer[
    (cancer['Rq'] > rq_cancer) &
    (cancer['Rdq'] > rdq_cancer) &
    (cancer['Circularity'] > circ_cancer)
]

filtered_df = pd.concat([benign_selected, cancer_selected], ignore_index=True)
print(f"✅ Imagini selectate pentru clasificare: {len(filtered_df)}")

# === 5. Pregătire date ===
X = filtered_df[['Rq', 'Rdq', 'Circularity']]
y = LabelEncoder().fit_transform(filtered_df['lesion_type'])

# Normalizare
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Împărțire în train/test (40% test set)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.4, stratify=y, random_state=42
)

# === 6. Clasificator Random Forest ===
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# === 7. Evaluare ===
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['benign', 'cancer'])

print(f"\n🌲 Accuracy: {round(accuracy * 100, 2)}%")
print("\n📊 Confusion Matrix:\n", conf_matrix)
print("\n🧾 Classification Report:\n", report)
