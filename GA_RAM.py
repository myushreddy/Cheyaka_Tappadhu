"""
GA-RAM: Genetic Algorithm with Rank-based Adaptive Mutation
Android Malware Detection - Baseline Classifier

This is the FIRST pipeline to run in your research.
It establishes the interpretable baseline using a single robust RF classifier.

Author: Based on your research methodology
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, classification_report, confusion_matrix)
import warnings
import time
warnings.filterwarnings('ignore')


# ============================================================================
# 1. GA-RAM CHROMOSOME CLASS
# ============================================================================

class GARAMChromosome:
    """
    Chromosome with Rank-based Adaptive Mutation capability
    Binary encoding: 1 = feature selected, 0 = feature not selected
    """
    
    def __init__(self, n_features):
        self.n_features = n_features
        self.genes = np.random.randint(0, 2, size=n_features)
        self.fitness = 0.0
        self.rank = 0  # Used for adaptive mutation
        
    def get_selected_features(self):
        """Return indices of genes that are 1 (selected features)"""
        return np.where(self.genes == 1)[0]
    
    def crossover(self, other, crossover_rate=0.8):
        """
        Single-point crossover between two parents
        Returns two children
        """
        if np.random.random() < crossover_rate:
            # Choose random crossover point
            point = np.random.randint(1, self.n_features)
            
            # Create children by swapping genes at crossover point
            child1_genes = np.concatenate([self.genes[:point], other.genes[point:]])
            child2_genes = np.concatenate([other.genes[:point], self.genes[point:]])
            
            # Create chromosome objects for children
            child1 = GARAMChromosome(self.n_features)
            child1.genes = child1_genes
            
            child2 = GARAMChromosome(self.n_features)
            child2.genes = child2_genes
            
            return child1, child2
        
        # No crossover occurred, return parents
        return self, other
    
    def mutate_ram(self, population_size, base_rate=0.01, alpha=2.0, effective_rank=None):
        """
        RANK-BASED ADAPTIVE MUTATION (RAM)
        
        Key innovation: mutation rate adapts based on chromosome fitness rank
        - Better chromosomes (high rank) → lower mutation rate (exploit)
        - Worse chromosomes (low rank) → higher mutation rate (explore)
        
        Formula: mutation_rate = base_rate × (1 + α × (1 - normalized_rank))
        
        Args:
            population_size: Size of population (for normalization)
            base_rate: Base mutation probability
            alpha: Adaptation strength (higher = more aggressive adaptation)
            effective_rank: Rank value to use for mutation (if None, uses self.rank)
        """
        # If no explicit rank is provided (e.g., for existing population),
        # fall back to the chromosome's own rank.
        if effective_rank is None:
            effective_rank = self.rank

        # Normalize rank: 0 (worst) to 1 (best)
        normalized_rank = effective_rank / population_size
        
        # Calculate adaptive mutation rate
        # High rank (good) → low mutation
        # Low rank (bad) → high mutation
        mutation_rate = base_rate * (1 + alpha * (1 - normalized_rank))
        
        # Apply bit-flip mutation
        for i in range(self.n_features):
            if np.random.random() < mutation_rate:
                self.genes[i] = 1 - self.genes[i]  # Flip bit
        
        # Constraint: Ensure at least 5 features selected
        # (prevents degenerate solutions)
        if np.sum(self.genes) < 5:
            random_indices = np.random.choice(self.n_features, 5, replace=False)
            self.genes[random_indices] = 1


# ============================================================================
# 2. FITNESS FUNCTION
# ============================================================================

def calculate_fitness_ram(chromosome, X_train, y_train, X_val, y_val, 
                          fast_mode=True, sample_size=5000, sample_indices=None):
    """
    Calculate fitness for GA-RAM
    
    Fitness = 0.7 × Accuracy + 0.3 × Feature_Reduction
    
    Components:
    1. Accuracy: Classification performance on validation set
    2. Feature Reduction: Penalty for using too many features
    
    Args:
        fast_mode: If True, use fewer estimators and sample data for faster evaluation
        sample_size: Number of samples to use for training in fast_mode
        sample_indices: Optional precomputed indices for training subset. When
                        provided, ALL chromosomes in a generation share the
                        same subset, ensuring fair fitness comparison.
    
    Returns: fitness score between 0 and 1
    """
    selected_features = chromosome.get_selected_features()
    
    # Edge case: no features selected
    if len(selected_features) == 0:
        return 0.0
    
    try:
        # Extract selected features
        X_train_selected = X_train[:, selected_features]
        X_val_selected = X_val[:, selected_features]
        
        # For faster evaluation, use a shared sample of training data.
        # To keep fitness comparable across chromosomes within the same
        # generation, sampling indices should be precomputed once per
        # generation and passed via `sample_indices`.
        if fast_mode and sample_indices is not None:
            X_train_subset = X_train_selected[sample_indices]
            y_train_subset = y_train[sample_indices]
        elif fast_mode and len(y_train) > sample_size:
            # Fallback path if no indices are provided; this should normally
            # not be used by the GA loop, but keeps the function robust.
            indices = np.random.choice(len(y_train), sample_size, replace=False)
            X_train_subset = X_train_selected[indices]
            y_train_subset = y_train[indices]
        else:
            X_train_subset = X_train_selected
            y_train_subset = y_train
        
        # Train Random Forest classifier with optimized parameters for speed
        n_estimators = 20 if fast_mode else 100  # Fewer trees for faster evaluation
        max_depth = 10 if fast_mode else 15
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        rf.fit(X_train_subset, y_train_subset)
        
        # Evaluate on validation set
        y_pred = rf.predict(X_val_selected)
        accuracy = accuracy_score(y_val, y_pred)
        
        # Feature reduction component
        feature_ratio = len(selected_features) / chromosome.n_features
        feature_bonus = 1 - feature_ratio  # Fewer features = higher bonus
        
        # Weighted combination
        fitness = (0.7 * accuracy) + (0.3 * feature_bonus)
        
        return fitness
    
    except Exception as e:
        print(f"⚠️  Error in fitness calculation: {e}")
        return 0.0


# ============================================================================
# 3. GA-RAM MAIN CLASS
# ============================================================================

class GARAM:
    """
    Genetic Algorithm with Rank-based Adaptive Mutation
    
    This is your baseline classifier that establishes interpretable
    feature selection before moving to ensemble methods.
    """
    
    def __init__(self, n_features, population_size=50, n_generations=30,
                 base_mutation_rate=0.01, alpha=2.0, tournament_size=3):
        """
        Initialize GA-RAM
        
        Args:
            n_features: Number of features in dataset
            population_size: Number of chromosomes in population
            n_generations: Number of generations to evolve
            base_mutation_rate: Base probability for mutation
            alpha: Adaptation strength for RAM
            tournament_size: Size of tournament for selection
        """
        self.n_features = n_features
        self.population_size = population_size
        self.n_generations = n_generations
        self.base_mutation_rate = base_mutation_rate
        self.alpha = alpha
        self.tournament_size = tournament_size
        
        self.population = []
        self.best_chromosome = None
        self.best_fitness = 0.0
        
        # History tracking
        self.fitness_history = []
        self.feature_count_history = []
        self.best_fitness_history = []
        
    def initialize_population(self):
        """Create initial random population"""
        self.population = [GARAMChromosome(self.n_features) 
                          for _ in range(self.population_size)]
        print(f"✓ Initialized population of {self.population_size} chromosomes")
    
    def evaluate_population(self, X_train, y_train, X_val, y_val, 
                            fast_mode=True, sample_size=5000):
        """
        Evaluate fitness for all chromosomes
        Updates best chromosome if better solution found.

        To ensure fair comparison across chromosomes in the SAME generation
        when `fast_mode=True`, we precompute a single subset of training
        indices and reuse it for every chromosome.
        """
        # Precompute a shared subset of training indices (if using fast mode)
        if fast_mode and len(y_train) > sample_size:
            sample_indices = np.random.choice(len(y_train), sample_size, replace=False)
        else:
            sample_indices = None

        for idx, chromosome in enumerate(self.population):
            chromosome.fitness = calculate_fitness_ram(
                chromosome, X_train, y_train, X_val, y_val,
                fast_mode=fast_mode,
                sample_size=sample_size,
                sample_indices=sample_indices,
            )
            
            # Track best chromosome across all generations
            if chromosome.fitness > self.best_fitness:
                self.best_fitness = chromosome.fitness
                self.best_chromosome = GARAMChromosome(self.n_features)
                self.best_chromosome.genes = chromosome.genes.copy()
                self.best_chromosome.fitness = chromosome.fitness
            
            # Progress indicator
            if (idx + 1) % 10 == 0:
                print(f"  Evaluated {idx + 1}/{len(self.population)} chromosomes...", end='\r')
    
    def rank_population(self):
        """
        Sort population by fitness and assign ranks
        
        Rank 1 = worst fitness
        Rank N = best fitness
        
        This ranking is used for adaptive mutation
        """
        # Sort ascending by fitness
        sorted_population = sorted(self.population, key=lambda x: x.fitness)
        
        # Assign ranks (1 to population_size)
        for rank, chromosome in enumerate(sorted_population, start=1):
            chromosome.rank = rank
    
    def tournament_selection(self):
        """
        Tournament selection for parent selection
        
        Randomly pick tournament_size chromosomes and return the best
        """
        tournament = np.random.choice(
            self.population, 
            self.tournament_size, 
            replace=False
        )
        return max(tournament, key=lambda x: x.fitness)
    
    def evolve(self, X_train, y_train, X_val, y_val):
        """
        Main evolutionary loop
        
        Returns: best chromosome found
        """
        print("\n" + "="*70)
        print("🧬 STARTING GA-RAM EVOLUTION")
        print("="*70)
        print(f"Population Size: {self.population_size}")
        print(f"Generations: {self.n_generations}")
        print(f"Base Mutation Rate: {self.base_mutation_rate}")
        print(f"Alpha (Adaptation): {self.alpha}")
        print(f"Tournament Size: {self.tournament_size}")
        print(f"\n⚡ OPTIMIZATION MODE: Fast evaluation enabled")
        print(f"   - Using 20 trees (instead of 100) for fitness evaluation")
        print(f"   - Sampling 5,000 training samples per evaluation")
        print(f"   - Final classifier will use full parameters")
        
        # Initialize population
        self.initialize_population()
        
        # Evolution loop
        generation_start_time = time.time()
        for generation in range(self.n_generations):
            gen_start = time.time()
            print(f"\n{'─'*70}")
            print(f"🔄 Generation {generation + 1}/{self.n_generations}")
            print(f"{'─'*70}")
            
            # Step 1: Evaluate fitness (use fast_mode=True for faster evaluation)
            # NOTE: evaluate_population ensures that all chromosomes in this
            # generation share the SAME training subset when fast_mode=True.
            self.evaluate_population(
                X_train, y_train, X_val, y_val,
                fast_mode=True,
                sample_size=5000,
            )
            print()  # New line after progress indicator
            
            # Step 2: Rank population (for adaptive mutation)
            self.rank_population()
            
            # Step 3: Track statistics
            avg_fitness = np.mean([c.fitness for c in self.population])
            avg_features = np.mean([np.sum(c.genes) for c in self.population])
            best_features = np.sum(self.best_chromosome.genes)
            
            self.fitness_history.append(avg_fitness)
            self.feature_count_history.append(avg_features)
            self.best_fitness_history.append(self.best_fitness)
            
            # Print generation stats
            print(f"📊 Best Fitness:      {self.best_fitness:.4f}")
            print(f"📊 Average Fitness:   {avg_fitness:.4f}")
            print(f"📊 Best Features:     {best_features}/{self.n_features}")
            print(f"📊 Average Features:  {avg_features:.1f}")
            
            # Step 4: Create next generation
            new_population = []
            
            # Elitism: Keep top 10% of population
            elite_size = int(0.1 * self.population_size)
            elite = sorted(self.population, key=lambda x: x.fitness, 
                          reverse=True)[:elite_size]
            new_population.extend(elite)
            
            # Generate offspring to fill rest of population
            offspring_count = 0
            while len(new_population) < self.population_size:
                # Parent selection via tournament
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                # Crossover
                child1, child2 = parent1.crossover(parent2, crossover_rate=0.8)
                
                # Rank-based Adaptive Mutation
                # Newborn children do not yet have meaningful ranks. To make
                # RAM effective immediately, we derive an "effective rank"
                # for each child from its parents (use the average rank).
                parent_avg_rank = (parent1.rank + parent2.rank) / 2.0

                child1.mutate_ram(
                    self.population_size,
                    self.base_mutation_rate,
                    self.alpha,
                    effective_rank=parent_avg_rank,
                )
                child2.mutate_ram(
                    self.population_size,
                    self.base_mutation_rate,
                    self.alpha,
                    effective_rank=parent_avg_rank,
                )
                
                new_population.extend([child1, child2])
                offspring_count += 2
            
            # Update population
            self.population = new_population[:self.population_size]
            gen_time = time.time() - gen_start
            elapsed_total = time.time() - generation_start_time
            avg_time_per_gen = elapsed_total / (generation + 1)
            estimated_remaining = avg_time_per_gen * (self.n_generations - generation - 1)
            print(f"✓ Preserved {elite_size} elite | Generated {offspring_count} offspring")
            print(f"⏱️  Generation time: {gen_time:.1f}s | Est. remaining: {estimated_remaining/60:.1f} min")
        
        # Final summary
        print("\n" + "="*70)
        print("✅ GA-RAM EVOLUTION COMPLETE")
        print("="*70)
        print(f"🏆 Best Fitness Achieved: {self.best_fitness:.4f}")
        print(f"🏆 Features Selected: {np.sum(self.best_chromosome.genes)}/{self.n_features}")
        print(f"🏆 Reduction: {(1-np.sum(self.best_chromosome.genes)/self.n_features)*100:.1f}%")
        
        return self.best_chromosome


# ============================================================================
# 4. FEATURE REFINEMENT
# ============================================================================

def refine_features_with_rf(X_train, y_train, selected_features, top_k=None):
    """
    Feature Refinement using Random Forest feature importance
    
    After GA selects features, RF ranks them by importance.
    Optionally keeps only top-k most important features.
    
    Args:
        X_train: Training data
        y_train: Training labels
        selected_features: Features selected by GA
        top_k: Number of top features to keep (None = keep all)
    
    Returns:
        refined_features: Final feature set
        importances: Feature importance scores
    """
    X_train_selected = X_train[:, selected_features]
    
    # Train Random Forest on selected features
    print("🔧 Training RF for feature refinement...")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_selected, y_train)
    
    # Get feature importances
    importances = rf.feature_importances_
    
    # Sort by importance (descending)
    importance_indices = np.argsort(importances)[::-1]
    
    if top_k is not None and top_k < len(selected_features):
        # Keep only top-k features
        top_indices = importance_indices[:top_k]
        refined_features = selected_features[top_indices]
        print(f"✓ Refined: {len(selected_features)} → {len(refined_features)} features")
    else:
        refined_features = selected_features
        print(f"✓ Kept all {len(refined_features)} GA-selected features")
    
    return refined_features, importances


# ============================================================================
# 5. COMPLETE PIPELINE
# ============================================================================

def run_garam_pipeline(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Complete GA-RAM pipeline for Android malware detection
    
    Pipeline stages:
    1. GA-RAM feature selection
    2. Feature refinement with RF
    3. Train final classifier
    4. Evaluate on test set
    
    Returns: dictionary with results
    """
    n_features = X_train.shape[1]
    
    print("\n" + "="*70)
    print("🚀 GA-RAM PIPELINE - ANDROID MALWARE DETECTION")
    print("="*70)
    print(f"Dataset Information:")
    print(f"  Total Features: {n_features}")
    print(f"  Training Samples: {len(y_train)}")
    print(f"  Validation Samples: {len(y_val)}")
    print(f"  Test Samples: {len(y_test)}")
    print(f"  Class Distribution: Benign={np.sum(y_train==0)}, Malware={np.sum(y_train==1)}")
    
    # ========================================================================
    # PHASE 1: GA-RAM FEATURE SELECTION
    # ========================================================================
    print("\n" + "="*70)
    print("📍 PHASE 1: GA-RAM FEATURE SELECTION")
    print("="*70)
    
    # Optimized parameters for faster execution with larger dataset
    garam = GARAM(
        n_features=n_features,
        population_size=30,  # Reduced from 50 for faster execution
        n_generations=20,   # Reduced from 30 for faster execution
        base_mutation_rate=0.01,
        alpha=2.0,
        tournament_size=3
    )
    
    best_chromosome = garam.evolve(X_train, y_train, X_val, y_val)
    selected_features = best_chromosome.get_selected_features()
    
    print(f"\n✅ Phase 1 Complete!")
    print(f"   Selected Features: {len(selected_features)}/{n_features}")
    print(f"   Reduction: {(1 - len(selected_features)/n_features)*100:.1f}%")
    
    # ========================================================================
    # PHASE 2: FEATURE REFINEMENT
    # ========================================================================
    print("\n" + "="*70)
    print("📍 PHASE 2: FEATURE REFINEMENT WITH RANDOM FOREST")
    print("="*70)
    
    # Optional: Can set top_k to further reduce features
    # For baseline, we keep all GA-selected features
    refined_features, importances = refine_features_with_rf(
        X_train, y_train, selected_features, top_k=None
    )
    
    print(f"\n✅ Phase 2 Complete!")
    print(f"   Final Features: {len(refined_features)}")
    
    # ========================================================================
    # PHASE 3: TRAIN FINAL CLASSIFIER
    # ========================================================================
    print("\n" + "="*70)
    print("📍 PHASE 3: TRAIN FINAL RANDOM FOREST CLASSIFIER")
    print("="*70)
    
    X_train_final = X_train[:, refined_features]
    X_test_final = X_test[:, refined_features]
    
    final_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("🔧 Training final classifier...")
    final_rf.fit(X_train_final, y_train)
    print("✓ Training complete")
    
    print(f"\n✅ Phase 3 Complete!")
    
    # ========================================================================
    # PHASE 4: EVALUATION ON TEST SET
    # ========================================================================
    print("\n" + "="*70)
    print("📍 PHASE 4: EVALUATION ON TEST SET")
    print("="*70)
    
    y_pred = final_rf.predict(X_test_final)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print("\n" + "="*70)
    print("📊 FINAL RESULTS - GA-RAM BASELINE")
    print("="*70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Benign', 'Malware']))
    
    print("\n🔢 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Benign  Malware")
    print(f"Actual Benign    {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"       Malware   {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    print(f"\n✅ Phase 4 Complete!")
    
    # Return all results
    return {
        'garam': garam,
        'selected_features': selected_features,
        'refined_features': refined_features,
        'feature_importances': importances,
        'final_classifier': final_rf,
        'y_pred': y_pred,
        'y_test': y_test,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }


# ============================================================================
# 6. VISUALIZATION
# ============================================================================

def visualize_garam_results(results):
    """Create comprehensive visualization of GA-RAM results"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Fitness Evolution (Large)
    ax1 = fig.add_subplot(gs[0, :2])
    generations = range(1, len(results['garam'].fitness_history) + 1)
    ax1.plot(generations, results['garam'].best_fitness_history, 
            marker='o', linewidth=2.5, markersize=6, 
            color='#2ecc71', label='Best Fitness')
    ax1.plot(generations, results['garam'].fitness_history, 
            marker='s', linewidth=2, markersize=4, 
            color='#3498db', alpha=0.7, label='Average Fitness')
    ax1.set_xlabel('Generation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fitness', fontsize=12, fontweight='bold')
    ax1.set_title('GA-RAM Fitness Evolution', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.0])
    
    # Plot 2: Feature Count Evolution
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(generations, results['garam'].feature_count_history, 
            marker='D', linewidth=2, markersize=5, color='#e74c3c')
    ax2.set_xlabel('Generation', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Avg Features', fontsize=11, fontweight='bold')
    ax2.set_title('Feature Selection', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Feature Selection Heatmap
    ax3 = fig.add_subplot(gs[1, :])
    n_features = results['garam'].n_features
    selected_mask = np.zeros(n_features)
    selected_mask[results['refined_features']] = 1
    
    im = ax3.imshow(selected_mask.reshape(1, -1), 
                   cmap='RdYlGn', aspect='auto', interpolation='nearest')
    ax3.set_xlabel('Feature Index', fontsize=12, fontweight='bold')
    ax3.set_title(f'Selected Features: {len(results["refined_features"])}/{n_features} ' +
                 f'({(len(results["refined_features"])/n_features)*100:.1f}%)', 
                 fontsize=14, fontweight='bold')
    ax3.set_yticks([])
    cbar = plt.colorbar(im, ax=ax3, orientation='horizontal', pad=0.1)
    cbar.set_label('Feature Status', fontsize=10, fontweight='bold')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Not Selected', 'Selected'])
    
    # Plot 4: Confusion Matrix
    ax4 = fig.add_subplot(gs[2, 0])
    cm = results['confusion_matrix']
    im = ax4.imshow(cm, cmap='Blues', interpolation='nearest')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Benign', 'Malware'], fontsize=10)
    ax4.set_yticklabels(['Benign', 'Malware'], fontsize=10)
    ax4.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax4.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax4.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax4.text(j, i, f'{cm[i, j]}', 
                    ha="center", va="center",
                    color=text_color, fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    
    # Plot 5: Performance Metrics
    ax5 = fig.add_subplot(gs[2, 1])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [results['accuracy'], results['precision'], 
              results['recall'], results['f1_score']]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    bars = ax5.barh(metrics, values, color=colors, alpha=0.8)
    ax5.set_xlim([0, 1.0])
    ax5.set_xlabel('Score', fontsize=11, fontweight='bold')
    ax5.set_title('Performance Metrics', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax5.text(value + 0.02, i, f'{value:.4f}', 
                va='center', fontsize=10, fontweight='bold')
    
    # Plot 6: Summary Stats
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')
    
    summary_text = f"""
    GA-RAM SUMMARY
    {'─'*25}
    
    Features:
      Original: {n_features}
      Selected: {len(results['refined_features'])}
      Reduction: {(1-len(results['refined_features'])/n_features)*100:.1f}%
    
    Performance:
      Accuracy: {results['accuracy']:.4f}
      F1-Score: {results['f1_score']:.4f}
    
    GA Parameters:
      Generations: {results['garam'].n_generations}
      Population: {results['garam'].population_size}
      Best Fitness: {results['garam'].best_fitness:.4f}
    """
    
    ax6.text(0.1, 0.5, summary_text, 
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('GA-RAM: Android Malware Detection Results', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('garam_results.png', dpi=300, bbox_inches='tight')
    print("\n✅ Visualization saved: garam_results.png")
    plt.show()


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("🔬 GA-RAM: GENETIC ALGORITHM WITH RANK-BASED ADAPTIVE MUTATION")
    print("   Android Malware Detection - Baseline Classifier")
    print("="*70)
    
    # Load data
    print("\n📁 Loading data...")
    data_path = Path(__file__).resolve().parent / 'data' / 'data_sample_25k.csv'
    labels_path = Path(__file__).resolve().parent / 'data' / 'y_labels.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    df = pd.read_csv(str(data_path))
    df_labels = pd.read_csv(str(labels_path))
    
    # Skip first 3 columns (SHA256, NOME, PACOTE) - these are metadata
    # Use columns 3 onwards as features
    X = df.iloc[:, 3:].values
    y = df_labels.iloc[:, 0].values
    
    # Labels are already 0 (benign) and 1 (malware) - no conversion needed
    
    print(f"✓ Dataset loaded successfully")
    print(f"  Shape: {X.shape}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Benign: {np.sum(y==0)}, Malware: {np.sum(y==1)}")
    
    # Split data
    print("\n📊 Splitting dataset...")
    
    # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split train: 80% train, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Validation set: {X_val.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # Run GA-RAM pipeline
    print("\n🚀 Starting GA-RAM pipeline...")
    results = run_garam_pipeline(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Visualize results
    print("\n📊 Generating comprehensive visualizations...")
    visualize_garam_results(results)
    
    # Final summary
    print("\n" + "="*70)
    print("✅ GA-RAM PIPELINE COMPLETE!")
    print("="*70)
    print(f"\n🎯 BASELINE RESULTS:")
    print(f"   Accuracy:       {results['accuracy']:.4f}")
    print(f"   Precision:      {results['precision']:.4f}")
    print(f"   Recall:         {results['recall']:.4f}")
    print(f"   F1-Score:       {results['f1_score']:.4f}")
    print(f"   Features Used:  {len(results['refined_features'])}/{X.shape[1]}")
    print(f"   Reduction:      {(1-len(results['refined_features'])/X.shape[1])*100:.1f}%")
    
    print("\n📌 Next Steps:")
    print("   1. ✅ GA-RAM baseline established")
    print("   2. → Run GA Stacking/MD for ensemble approach")
    print("   3. → Run DL ATIDS for deep learning comparison")
    
    print("\n" + "="*70)