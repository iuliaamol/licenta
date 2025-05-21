import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# Step 1: Load data
df = pd.read_csv('../evaluation/ellipse_features2.csv')

# Step 2: Split into benign and malignant
benign = df[df['label'] == 'benign']
malignant = df[df['label'] == 'cancer']

# Step 3: List of features (excluding filename, label, density)
features = df.columns.drop(['filename', 'label', 'density'])

# Step 4: Perform Mann-Whitney U-test for each feature
results = []

for feature in features:
    stat, p = mannwhitneyu(benign[feature], malignant[feature], alternative='two-sided')
    results.append((feature, p))

# Step 5: Create a DataFrame with the results
results_df = pd.DataFrame(results, columns=['Feature', 'p-value'])
results_df = results_df.sort_values('p-value')

# Step 6: Display top distinguishing features
print(results_df.head(10))

# Optional: Visualize top 5 features
top_features = results_df['Feature'].head(5)

for feature in top_features:
    plt.figure()
    sns.boxplot(x='label', y=feature, data=df)
    plt.title(f'Distribution of {feature} by Label')
    plt.show()
