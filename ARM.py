import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import mutual_info_classif

# ==========================================
# 1. SETUP & PATHS
# ==========================================
# We define the base directory so we don't have to type it every time
BASE_DIR = r"D:\ARM"
INPUT_FILE = os.path.join(BASE_DIR, "data_sample_25k.csv")

print(f"Working Directory: {BASE_DIR}")

# Check if input file exists
if not os.path.exists(INPUT_FILE):
    print(f" ERROR: File not found at {INPUT_FILE}")
    print("Please check the file name and location.")
    exit()

# ==========================================
# 2. LOAD & INSPECT DATA
# ==========================================
print("\nLoading dataset...")
df = pd.read_csv(INPUT_FILE)

print("First 5 rows:")
print(df.head())
print("\nShape:", df.shape)

# Check for label columns
print("\nChecking for potential label columns...")
found_label = False
for col in df.columns:
    if 'label' in col.lower() or 'class' in col.lower() or 'malware' in col.lower():
        print(f" -> Found potential label: {col}")
        found_label = True

if not found_label:
    print(" -> No explicit label column found.")

# ==========================================
# 3. CLEAN & SEPARATE FEATURES
# ==========================================
# Columns that are NOT features (Metadata)
non_feature_cols = ['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'class', 'label']

# Keep only real features (Drop columns if they exist)
# errors='ignore' ensures it doesn't crash if a column is missing
X = df.drop(columns=non_feature_cols, errors='ignore')

print("\nFeature matrix shape:", X.shape)
# print(X.head()) # Optional: Uncomment to see data

# Check unique values (To ensure it's binary 0/1)
unique_vals = np.unique(X.values)
print("Unique values in X:", unique_vals)

# Save Cleaned Features locally
x_save_path = os.path.join(BASE_DIR, "X_features_clean.csv")
X.to_csv(x_save_path, index=False)
print(f" Saved clean features to: {x_save_path}")

# ==========================================
# 4. CREATE PLACEHOLDER LABELS
# ==========================================
# Create a placeholder label column (all -1 for now)
# NOTE: In a real run, you would extract real labels here.
y = pd.Series([-1] * len(X), name="label")

y_save_path = os.path.join(BASE_DIR, "y_labels_placeholder.csv")
y.to_csv(y_save_path, index=False)
print(f" Saved placeholder labels to: {y_save_path}")

# ==========================================
# 5. COMBINE & SAVE (PIPELINE READY)
# ==========================================
# Combine features + placeholder labels into one dataframe
df_full = pd.concat([X, y], axis=1)

full_save_path = os.path.join(BASE_DIR, "XY_placeholder.csv")
df_full.to_csv(full_save_path, index=False)

print(f" Combined dataset saved to: {full_save_path}")
print("Combined Shape:", df_full.shape)

# ==========================================
# 6. MUTUAL INFORMATION (TEST)
# ==========================================
print("\nRunning Mutual Information Test...")

# CRITICAL FIX for VS Code:
# mutual_info_classif REQUIRES at least 2 classes (e.g., 0 and 1).
# Since our 'y' is all -1, the code will crash or return 0.
# We create a specific 'y_demo' just to test if the code works.

if y.nunique() <= 1:
    print(" Warning: Labels are all same (-1). Generating RANDOM labels just to test the MI code...")
    y_demo = np.random.randint(0, 2, len(y))
else:
    y_demo = y

# Calculate MI Scores
mi = mutual_info_classif(X, y_demo, discrete_features=True)

# Create a dataframe with feature names + MI scores
mi_df = pd.DataFrame({
    'Feature': X.columns,
    'MI_score': mi
})

# Sort descending
mi_df = mi_df.sort_values(by='MI_score', ascending=False)

mi_save_path = os.path.join(BASE_DIR, "MI_scores_result.csv")
mi_df.to_csv(mi_save_path, index=False)

print(f" MI scores saved to: {mi_save_path}")
print("Top 5 Features by MI Score:")
print(mi_df.head(5))

# ==========================================
# 7. GA-RAM POPULATION INIT (TEST)
# ==========================================
print("\nInitializing GA-RAM Population...")

n_features = X.shape[1]        # Number of columns
population_size = 10           # Small for test
generations = 5                # Testing only
mutation_rate = 0.2            

# Initialize population (Random binary masks)
population = np.random.randint(0, 2, size=(population_size, n_features))

print(f"GA-RAM initialized with shape: {population.shape}")
print("First individual (binary mask):")
print(population[0])


