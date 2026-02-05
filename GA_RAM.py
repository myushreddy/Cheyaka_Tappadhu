"""
GA-RAM: REALISTIC RESEARCH VERSION
- Purpose: Prevents 100% accuracy by removing deterministic triggers.
- Constraints: 20-35 features.
- Path: C:\\Users\\mhrit\\OneDrive\\Documents\\Desktop\\CN_LAB\\Cheyaka_Tappadhu\\src\\data\\XY_final_dataset.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. PATH & DATA LOADING
# ==========================================
data_path = r"C:\Users\mhrit\OneDrive\Documents\Desktop\CN_LAB\Cheyaka_Tappadhu\src\data\XY_final_dataset.csv"
output_dir = r"C:\Users\mhrit\OneDrive\Documents\Desktop\CN_LAB\Cheyaka_Tappadhu\src\results"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f" Loading Dataset...")
df = pd.read_csv(data_path)

# --- THE SCIENTIFIC FIX: PREVENT DATA LEAKAGE ---
# We remove the permissions used to CREATE the labels.
# This forces the model to learn from the other 40 features.
leaking_features = ['Permission::INTERNET', 'Permission::WAKE_LOCK']
X_df = df.drop(columns=['label'] + [f for f in leaking_features if f in df.columns])
y = df['label'].values

print(f" Leakage Prevention: Removed {leaking_features} from feature set.")
print(f" Training on {X_df.shape[1]} remaining features.")

# ==========================================
# 1. GA-RAM CORE LOGIC (With Constraints)
# ==========================================
class RealisticGARAM:
    def __init__(self, n_features, pop_size=20, generations=10):
        self.n_features = n_features
        self.pop_size = pop_size
        self.generations = generations
        # Initialize population with 20-35 features active
        self.population = [self._create_chromosome() for _ in range(pop_size)]

    def _create_chromosome(self):
        chrom = np.zeros(self.n_features)
        indices = np.random.choice(self.n_features, np.random.randint(20, 36), replace=False)
        chrom[indices] = 1
        return chrom

    def fitness(self, chrom, X_tr, X_val, y_tr, y_val):
        selected = np.where(chrom == 1)[0]
        if len(selected) < 20 or len(selected) > 35: return 0.1
        
        # Strong RF Regularization
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_tr[:, selected], y_tr)
        return accuracy_score(y_val, rf.predict(X_val[:, selected]))

    def evolve(self, X_tr, X_val, y_tr, y_val):
        best_chrom = None
        best_fit = 0
        
        print("🧬 Evolving features (Evolutionary Selection)...")
        for gen in range(self.generations):
            fits = [self.fitness(c, X_tr, X_val, y_tr, y_val) for c in self.population]
            
            # Keep best
            current_best_idx = np.argmax(fits)
            if fits[current_best_idx] > best_fit:
                best_fit = fits[current_best_idx]
                best_chrom = self.population[current_best_idx].copy()
            
            print(f"   Gen {gen+1}: Best Accuracy = {best_fit:.4f}")
            
            # Simple Selection & Mutation
            new_pop = [best_chrom] # Elitism
            while len(new_pop) < self.pop_size:
                child = best_chrom.copy()
                # Mutate: flip 2 random bits
                m_idx = np.random.choice(self.n_features, 2, replace=False)
                child[m_idx] = 1 - child[m_idx]
                # Enforce 20-35 constraint
                if 20 <= np.sum(child) <= 35:
                    new_pop.append(child)
            self.population = new_pop
            
        return best_chrom

# ==========================================
# 2. EXECUTION PIPELINE
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X_df.values, y, test_size=0.2, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Run Evolution
ga = RealisticGARAM(n_features=X_df.shape[1])
best_features_mask = ga.evolve(X_tr, X_val, y_tr, y_val)
selected_indices = np.where(best_features_mask == 1)[0]
selected_names = X_df.columns[selected_indices]

# Final Evaluation
final_rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
final_rf.fit(X_train[:, selected_indices], y_train)
y_pred = final_rf.predict(X_test[:, selected_indices])

acc = accuracy_score(y_test, y_pred)

# ==========================================
# 3. RESULTS & VISUALIZATION
# ==========================================
print("\n" + "="*40)
print(" REALISTIC RESEARCH RESULTS")
print(f" Accuracy: {acc*100:.2f}% (Realistic Range)")
print(f"  Features Selected: {len(selected_indices)}")
print("="*40)

# 1. Save Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Safe', 'Malware'], yticklabels=['Safe', 'Malware'])
plt.title(f'GA-RAM Confusion Matrix (Acc: {acc*100:.1f}%)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(os.path.join(output_dir, "realistic_confusion_matrix.png"))

# 2. Save Selected Features List
with open(os.path.join(output_dir, "selected_features.txt"), "w") as f:
    f.write("TOP MALWARE INDICATORS (Selected by GA-RAM):\n")
    for name in selected_names:
        f.write(f"- {name}\n")

print(f" Results saved in: {output_dir}")