import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

# ==========================================
# 1. SETUP & PATHS
# ==========================================
# Get project root directory (one level up from src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, "data_sample_25k.csv")

print(f"Project Root: {PROJECT_ROOT}")
print(f"Data Directory: {DATA_DIR}")

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
print(f"Actual number of features: {X.shape[1]}")
# print(X.head()) # Optional: Uncomment to see data

# Check unique values (To ensure it's binary 0/1)
unique_vals = np.unique(X.values)
print("Unique values in X:", unique_vals)

# Save Cleaned Features locally
x_save_path = os.path.join(DATA_DIR, "X_features_clean.csv")
X.to_csv(x_save_path, index=False)
print(f" Saved clean features to: {x_save_path}")

# ==========================================
# 4. CREATE PLACEHOLDER LABELS
# ==========================================
# Create a placeholder label column (all -1 for now)
# NOTE: In a real run, you would extract real labels here.
y = pd.Series([-1] * len(X), name="label")

y_save_path = os.path.join(DATA_DIR, "y_labels_placeholder.csv")
y.to_csv(y_save_path, index=False)
print(f" Saved placeholder labels to: {y_save_path}")

# ==========================================
# 5. COMBINE & SAVE (PIPELINE READY)
# ==========================================
# Combine features + placeholder labels into one dataframe
df_full = pd.concat([X, y], axis=1)

full_save_path = os.path.join(DATA_DIR, "XY_placeholder.csv")
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

# Debug: Print actual feature count
print(f"\n[DEBUG] Total features in X: {len(X.columns)}")
print(f"[DEBUG] Total MI scores calculated: {len(mi_df)}")

mi_save_path = os.path.join(DATA_DIR, "MI_scores_result.csv")
mi_df.to_csv(mi_save_path, index=False)

print(f" MI scores saved to: {mi_save_path}")
print("Top 5 Features by MI Score:")
print(mi_df.head(5))

# ==========================================
# 6.5 VISUALIZE BEFORE & AFTER MUTUAL INFORMATION
# ==========================================
print("\nGenerating Before & After Mutual Information visualizations...")

# Set style for better-looking plots
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'

# Create figure with 2 subplots side by side with proper spacing
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.1, wspace=0.35)

# Define threshold - keep top 42 features
top_n_after = 42
mi_after_exact = mi_df.head(top_n_after)
threshold_value = mi_after_exact['MI_score'].min()

# Get all actual features from the dataset
all_features_exact = mi_df.sort_values(by='MI_score', ascending=False)

print(f"\nACTUAL FEATURES FROM data_sample_25k.csv:")
print(f"Total features in dataset: {len(all_features_exact)}")
print(f"Feature names: {list(all_features_exact['Feature'].values)}\n")

# ===== LEFT: BEFORE - All actual features =====
# Create colors: highlight selected vs noise
colors_before = ['#2ecc71' if score >= threshold_value else '#e74c3c' 
                 for score in all_features_exact['MI_score'].values]

axes[0].bar(range(len(all_features_exact)), all_features_exact['MI_score'].values, 
            color=colors_before, edgecolor='black', linewidth=0.3, alpha=0.85)
axes[0].axhline(y=threshold_value, color='orange', linestyle='--', linewidth=2.5, 
                label=f'Selection Threshold')
axes[0].set_xlabel('Features (from data_sample_25k)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Mutual Information Score', fontsize=12, fontweight='bold')
axes[0].set_title(f'BEFORE: {len(all_features_exact)} Actual Features\n(Green = Selected | Red = Noise)', 
                  fontsize=13, fontweight='bold', pad=15)
axes[0].grid(axis='y', alpha=0.3)
axes[0].legend(loc='upper right', fontsize=10)

# ===== RIGHT: AFTER - Top 42 actual selected features with names =====
mi_after_display = mi_df.head(top_n_after).sort_values(by='MI_score', ascending=False)

axes[1].barh(range(len(mi_after_display)), mi_after_display['MI_score'].values, 
             color='#2ecc71', edgecolor='darkgreen', linewidth=0.8, alpha=0.85)
axes[1].set_yticks(range(len(mi_after_display)))
axes[1].set_yticklabels(mi_after_display['Feature'].values, fontsize=8.5)
axes[1].set_xlabel('Mutual Information Score', fontsize=12, fontweight='bold')
axes[1].set_title(f'AFTER: Top {top_n_after} Selected Features\n({len(all_features_exact) - top_n_after} features removed)', 
                  fontsize=13, fontweight='bold', pad=15)
axes[1].grid(axis='x', alpha=0.3)
axes[1].invert_yaxis()

# Add a text box with summary
reduction_pct = ((len(all_features_exact) - top_n_after) / len(all_features_exact) * 100)
summary_text = f'Reduction: {len(all_features_exact)} → {top_n_after}\n({reduction_pct:.1f}% fewer features)'
axes[1].text(0.98, 0.02, summary_text, transform=axes[1].transAxes, 
             fontsize=11, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='#d5f4e6', alpha=0.9, 
                      edgecolor='#2ecc71', linewidth=2), fontweight='bold')

plt.suptitle('Mutual Information Feature Selection - BEFORE & AFTER', 
             fontsize=16, fontweight='bold', y=0.98)

# Save the figure to images folder
images_dir = os.path.join(PROJECT_ROOT, "images")
os.makedirs(images_dir, exist_ok=True)
viz_save_path = os.path.join(images_dir, "MI_before_after_visualization.png")
plt.savefig(viz_save_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f" Visualization saved to: {viz_save_path}")

# Display the plot
plt.show()

# Print summary statistics
print(f"\n{'='*80}")
print(f"FEATURE SELECTION SUMMARY - ACTUAL DATA")
print(f"{'='*80}")
print(f"Data source: data_sample_25k.csv")
print(f"Total features BEFORE:    {len(all_features_exact)}")
print(f"Total features AFTER:     {top_n_after}")
print(f"Features removed:         {len(all_features_exact) - top_n_after}")
print(f"Reduction:                {reduction_pct:.1f}%")
print(f"MI threshold:             {threshold_value:.6f}")
print(f"{'='*80}\n")

print(f"BEFORE - All {len(all_features_exact)} Actual Features from data_sample_25k:")
print(f"{'-'*80}")
for idx, (_, row) in enumerate(all_features_exact.iterrows(), 1):
    print(f"{idx:3d}. {row['Feature']:<25s} | MI Score: {row['MI_score']:.6f}")

print(f"\n{'='*80}\n")
print(f"AFTER - Top {top_n_after} Selected Features:")
print(f"{'-'*80}")
for idx, (_, row) in enumerate(mi_after_display.iterrows(), 1):
    print(f"{idx:3d}. {row['Feature']:<25s} | MI Score: {row['MI_score']:.6f}")
print(f"{'='*80}\n")



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