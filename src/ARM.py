import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_classif

# ==========================================
# 0. DIRECT PATH SETUP
# ==========================================
# This identifies your main project folder (Cheyaka_Tappadhu)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Path to the actual CSV file based on your input
source_file = os.path.join(parent_dir, "data", "data_sample_25k.csv")

# Path where we will save the results
output_dir = os.path.join(script_dir, "data")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(source_file):
    print(f" ERROR: File not found at: {source_file}")
    print("Please verify the file is inside the 'data' folder of your project.")
    exit()

print(f" Found and processing: {source_file}")

# ==========================================
# 1. LOAD & CLEAN FEATURES
# ==========================================
df = pd.read_csv(source_file)

# We skip metadata (first 5 columns: SHA256, NOME, PACOTE, API_MIN, API)
# We take the rest as binary features (permissions/API calls)
X = df.iloc[:, 5:] 
X.to_csv(os.path.join(output_dir, "X_features_clean.csv"), index=False)
print(f" Step 1: Extracted {X.shape[1]} features.")

# ==========================================a
# 2. GENERATE SUPERVISED LABELS
# ==========================================
print(" Step 2: Generating labels...")
# Rule: If app has INTERNET and WAKE_LOCK permissions, mark as Malware (1)
y_logic = np.where((X['Permission::INTERNET'] == 1) & (X['Permission::WAKE_LOCK'] == 1), 1, 0)
y = pd.DataFrame(y_logic, columns=['label'])
y.to_csv(os.path.join(output_dir, "y_labels.csv"), index=False)

# ==========================================
# 3. RANKING & SELECTING TOP 42
# ==========================================
print(" Step 3: Ranking features using Mutual Information (Please wait)...")
mi = mutual_info_classif(X, y['label'], discrete_features=True, random_state=42)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI_score': mi}).sort_values(by='MI_score', ascending=False)
mi_df.to_csv(os.path.join(output_dir, "MI_scores_result.csv"), index=False)

# Select and save top 42
top_42_names = mi_df.head(42)['Feature'].tolist()
X_optimized = X[top_42_names]
X_optimized.to_csv(os.path.join(output_dir, "X_features_optimized.csv"), index=False)

# ==========================================
# 4. FINAL MASTER DATASET
# ==========================================
xy_final = pd.concat([X_optimized, y], axis=1)
xy_final.to_csv(os.path.join(output_dir, "XY_final_dataset.csv"), index=False)

print("\n" + "="*40)
print(" SUCCESS! PROJECT DATA IS READY.")
print(f" Final Dataset: {xy_final.shape[1]} columns (42 features + 1 label)")
print(f" Results saved in: {output_dir}")
print("="*40)

# ==========================================
# 5. VISUALIZATION (BEFORE & AFTER PLOT)
# ==========================================
import matplotlib.pyplot as plt
import seaborn as sns

print(" Step 5: Generating visualization image...")

# Sort full MI scores
mi_df_sorted = mi_df.sort_values(by='MI_score', ascending=False).reset_index(drop=True)

# Selection threshold (42nd feature score)
threshold = mi_df_sorted.iloc[41]['MI_score']

# Split selected vs removed
selected = mi_df_sorted.iloc[:42]
removed = mi_df_sorted.iloc[42:]

# Create figure
plt.figure(figsize=(18, 8))
# Remove leakage features used to create label
leakage_features = ['Permission::INTERNET', 'Permission::WAKE_LOCK']
X_mi = X.drop(columns=leakage_features)

mi = mutual_info_classif(X_mi, y['label'], discrete_features=True, random_state=42)
mi_df = pd.DataFrame({'Feature': X_mi.columns, 'MI_score': mi}) \
            .sort_values(by='MI_score', ascending=False)

# ================= LEFT: BEFORE =================
plt.subplot(1, 2, 1)
colors = ['green' if i < 42 else 'red' for i in range(len(mi_df_sorted))]
plt.bar(range(len(mi_df_sorted)), mi_df_sorted['MI_score'], color=colors)
plt.axhline(y=threshold, color='orange', linestyle='--', label='Selection Threshold')
plt.title("BEFORE: 95 Actual Features\n(Green = Selected | Red = Noise)")
plt.xlabel("Features (from data_sample_25k)")
plt.ylabel("Mutual Information Score")
plt.legend()

# ================= RIGHT: AFTER =================
plt.subplot(1, 2, 2)
sns.barplot(
    x='MI_score',
    y='Feature',
    data=selected,
    palette='Greens_r'
)
plt.title("AFTER: Top 42 Selected Features\n(53 features removed)")
plt.xlabel("Mutual Information Score")
plt.ylabel("")

plt.tight_layout()

# Save image
image_path = os.path.join(output_dir, "Mutual_Information_Feature_Selection.png")
plt.savefig(image_path, dpi=300, bbox_inches='tight')
plt.close()

print(f" Visualization saved at: {image_path}")
