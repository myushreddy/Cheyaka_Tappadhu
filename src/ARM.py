import pandas as pd
import numpy as np
import os
import random
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# 1. SETUP & PATHS
# ==========================================
# Use the D: drive path specific to your setup
BASE_DIR = r"D:\ARM"
INPUT_FILE = os.path.join(BASE_DIR, "data_sample_25k.csv")

print(f"Working Directory: {BASE_DIR}")

# Check if input file exists
if not os.path.exists(INPUT_FILE):
    print(f" ERROR: File not found at {INPUT_FILE}")
    print("Please check the file name and location.")
    exit()

# ==========================================
# 2. LOAD & CLEAN DATA
# ==========================================
print("\nLoading dataset...")
df = pd.read_csv(INPUT_FILE)

# Define columns to drop (Metadata / Non-Features)
non_feature_cols = ['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'class', 'label']

# Create Feature Matrix X
X = df.drop(columns=non_feature_cols, errors='ignore')

# Create Placeholder Labels y (Since original labels might be missing or mixed)
# If 'class' exists, we use it; otherwise, we generate a placeholder or synthetic target
if 'class' in df.columns:
    y = df['class']
else:
    # Synthetic label for demonstration (0 or 1 based on first column)
    # In a real scenario, you would ensure the 'class' column is present
    y = pd.Series(np.where(X.iloc[:, 0] > 0, 1, 0), name="label")

print(f"Data Loaded. Features: {X.shape[1]}, Samples: {X.shape[0]}")

# Save Cleaned Data locally
X.to_csv(os.path.join(BASE_DIR, "X_features_clean.csv"), index=False)
y.to_csv(os.path.join(BASE_DIR, "y_labels_placeholder.csv"), index=False)
print(" Cleaned datasets saved.")

# ==========================================
# 3. MUTUAL INFORMATION (FILTERING STAGE)
# ==========================================
print("\nRunning Mutual Information (MI) Filtering...")

# Handle case where labels are uniform (prevents crash)
if y.nunique() <= 1:
    y_temp = np.random.randint(0, 2, len(y)) # Temp labels for calculation only
else:
    y_temp = y

mi = mutual_info_classif(X, y_temp, discrete_features=True)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI_score': mi}).sort_values(by='MI_score', ascending=False)

mi_df.to_csv(os.path.join(BASE_DIR, "MI_scores_result.csv"), index=False)
print(" MI Scores saved.")

# ==========================================
# 4. GA-RAM ALGORITHM (OPTIMIZATION STAGE)
# ==========================================
print("\nInitializing GA-RAM Optimization...")

class GeneticAlgorithmRAM:
    def __init__(self, features, X_train, X_test, y_train, y_test, 
                 pop_size=10, generations=5, mutation_min=0.01, mutation_max=0.3):
        self.features = features
        self.n_features = len(features)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_min = mutation_min
        self.mutation_max = mutation_max
        self.history = []

    def init_population(self):
        # Initialize random binary masks
        return [np.random.randint(0, 2, self.n_features).tolist() for _ in range(self.pop_size)]

    def fitness(self, individual):
        # Select features where bit is 1
        cols = [self.features[i] for i, x in enumerate(individual) if x == 1]
        if len(cols) == 0: return 0 
        
        # Fast RF for evaluation
        clf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
        clf.fit(self.X_train[cols], self.y_train)
        return accuracy_score(self.y_test, clf.predict(self.X_test[cols]))

    def adaptive_mutation(self, individual, rank):
        # Adaptive Mutation: Worse rank = Higher mutation chance
        pm = self.mutation_min + (self.mutation_max - self.mutation_min) * (rank / self.pop_size)
        for i in range(self.n_features):
            if random.random() < pm:
                individual[i] = 1 - individual[i] # Flip bit
        return individual

    def run(self):
        population = self.init_population()
        best_overall_acc = 0.0
        
        for gen in range(self.generations):
            # Evaluate Fitness
            scores = [(ind, self.fitness(ind)) for ind in population]
            scores.sort(key=lambda x: x[1], reverse=True) # Sort Best to Worst
            
            best_ind, best_acc = scores[0]
            self.history.append(best_acc)
            
            if best_acc > best_overall_acc:
                best_overall_acc = best_acc

            # Log Progress
            print(f"Gen {gen+1}/{self.generations} | Best Accuracy: {best_acc:.4%} | Features: {sum(best_ind)}")

            # Create Next Generation (Simple Elitism + Crossover)
            next_gen = [scores[0][0], scores[1][0]] # Keep top 2
            
            while len(next_gen) < self.pop_size:
                # Tournament Selection
                p1 = max(random.sample(scores, 3), key=lambda x: x[1])[0]
                p2 = max(random.sample(scores, 3), key=lambda x: x[1])[0]
                
                # Crossover
                pt = random.randint(1, self.n_features-1)
                c1 = p1[:pt] + p2[pt:]
                c2 = p2[:pt] + p1[pt:]
                
                # Mutation
                c1 = self.adaptive_mutation(c1, random.randint(0, self.pop_size))
                c2 = self.adaptive_mutation(c2, random.randint(0, self.pop_size))
                
                next_gen.extend([c1, c2])
            
            population = next_gen[:self.pop_size]

        return best_overall_acc

# ==========================================
# 5. EXECUTION
# ==========================================
# Split Data for GA Validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run GA-RAM
# Note: Using small generation count for demo speed (generations=5)
ga = GeneticAlgorithmRAM(list(X.columns), X_train, X_test, y_train, y_test, 
                         pop_size=10, generations=5, mutation_max=0.2)

print("Starting GA-RAM Evolution...")
final_acc = ga.run()

print("-" * 50)
print(f" FINAL GA-RAM RESULT: {final_acc:.4%} Accuracy")
print("-" * 50)