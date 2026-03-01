"""
Credit Card Fraud Detection Project
====================================
A complete ML pipeline for detecting fraudulent credit card transactions.

Author: Data Scientist
Dataset: Synthetic (similar to Kaggle Credit Card Fraud Detection)
"""

# ============================================================================
# IMPORTS
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CUSTOM SMOTE IMPLEMENTATION (to avoid version conflicts)
# ============================================================================
class SimpleSMOTE:
    """
    Simple implementation of SMOTE (Synthetic Minority Over-sampling Technique)
    for handling imbalanced datasets.
    """
    def __init__(self, sampling_strategy='auto', k_neighbors=5, random_state=42):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
    def fit_resample(self, X, y):
        np.random.seed(self.random_state)
        
        X = np.array(X)
        y = np.array(y)
        
        # Get class counts
        classes, counts = np.unique(y, return_counts=True)
        min_class = classes[np.argmin(counts)]
        maj_class = classes[np.argmax(counts)]
        
        # Number of samples to generate
        n_samples_to_generate = counts[maj_class] - counts[min_class]
        
        # Get minority class samples
        minority_samples = X[y == min_class]
        
        # Generate synthetic samples
        synthetic_samples = []
        for _ in range(n_samples_to_generate):
            # Randomly select a minority sample
            idx = np.random.randint(0, len(minority_samples))
            sample = minority_samples[idx]
            
            # Find k nearest neighbors
            distances = np.linalg.norm(minority_samples - sample, axis=1)
            k_neighbors_idx = np.argsort(distances)[1:self.k_neighbors+1]
            k_neighbors = minority_samples[k_neighbors_idx]
            
            # Randomly select a neighbor and interpolate
            neighbor = k_neighbors[np.random.randint(0, len(k_neighbors))]
            diff = neighbor - sample
            synthetic = sample + np.random.random() * diff
            synthetic_samples.append(synthetic)
        
        synthetic_samples = np.array(synthetic_samples)
        
        # Combine original and synthetic samples
        X_resampled = np.vstack([X, synthetic_samples])
        y_resampled = np.hstack([y, np.full(n_samples_to_generate, min_class)])
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X_resampled))
        
        return X_resampled[shuffle_idx], y_resampled[shuffle_idx]

# Create SMOTE instance
smote = SimpleSMOTE(random_state=42)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("CREDIT CARD FRAUD DETECTION PROJECT")
print("="*80)

# ============================================================================
# STEP 0: DATA GENERATION (Synthetic Dataset)
# ============================================================================
print("\n" + "="*80)
print("STEP 0: GENERATING SYNTHETIC DATASET")
print("="*80)

def generate_synthetic_credit_card_data(n_samples=100000, fraud_ratio=0.0017):
    """
    Generate synthetic credit card transaction data similar to Kaggle dataset.
    
    The Kaggle dataset has:
    - 28 PCA-transformed features (V1-V28)
    - Time (seconds elapsed from first transaction)
    - Amount (transaction amount)
    - Class (0 = legitimate, 1 = fraud)
    
    Args:
        n_samples: Total number of samples to generate
        fraud_ratio: Proportion of fraudulent transactions (default ~0.17%)
    
    Returns:
        DataFrame with synthetic credit card data
    """
    n_fraud = int(n_samples * fraud_ratio)
    n_legitimate = n_samples - n_fraud
    
    print(f"Generating {n_samples:,} samples...")
    print(f"  - Legitimate transactions: {n_legitimate:,}")
    print(f"  - Fraudulent transactions: {n_fraud:,}")
    
    # Generate legitimate transactions
    legitimate_data = {
        'Time': np.random.uniform(0, 172800, n_legitimate),  # 2 days in seconds
        'Amount': np.random.exponential(80, n_legitimate),  # Most transactions are small
    }
    
    # Generate V1-V28 features for legitimate (normal distribution)
    for i in range(1, 29):
        legitimate_data[f'V{i}'] = np.random.normal(0, 1, n_legitimate)
    
    # Generate fraudulent transactions
    fraud_data = {
        'Time': np.random.uniform(0, 172800, n_fraud),
        'Amount': np.random.exponential(150, n_fraud),  # Fraud amounts tend to be higher
    }
    
    # Generate V1-V28 features for fraud (slightly different distribution)
    for i in range(1, 29):
        fraud_data[f'V{i}'] = np.random.normal(0.5, 1.2, n_fraud)
    
    # Create DataFrames
    df_legitimate = pd.DataFrame(legitimate_data)
    df_legitimate['Class'] = 0
    
    df_fraud = pd.DataFrame(fraud_data)
    df_fraud['Class'] = 1
    
    # Combine and shuffle
    df = pd.concat([df_legitimate, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Class distribution:\n{df['Class'].value_counts()}")
    
    return df

# Generate the dataset
df = generate_synthetic_credit_card_data(n_samples=100000, fraud_ratio=0.0017)

print("\n✅ Synthetic dataset generated successfully!")
print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nDataset statistics:")
print(df.describe())

# Save the dataset
df.to_csv('creditcard_synthetic.csv', index=False)
print("\n✅ Dataset saved as 'creditcard_synthetic.csv'")

# ============================================================================
# STEP 1: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*80)
print("STEP 1: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80)

# 1.1 Class Distribution
print("\n--- 1.1 Class Distribution ---")
class_counts = df['Class'].value_counts()
print(f"Class Distribution:")
print(f"  Legitimate (0): {class_counts[0]:,} ({class_counts[0]/len(df)*100:.2f}%)")
print(f"  Fraudulent (1): {class_counts[1]:,} ({class_counts[1]/len(df)*100:.2f}%)")

# Create class distribution visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
colors = ['#2ecc71', '#e74c3c']
axes[0].bar(['Legitimate (0)', 'Fraudulent (1)'], 
            [class_counts[0], class_counts[1]], 
            color=colors, edgecolor='black', linewidth=1.5)
axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Transactions')
for i, v in enumerate([class_counts[0], class_counts[1]]):
    axes[0].text(i, v + 1000, f'{v:,}', ha='center', fontweight='bold')

# Pie chart
axes[1].pie([class_counts[0], class_counts[1]], 
            labels=['Legitimate', 'Fraudulent'],
            colors=colors,
            autopct='%1.3f%%',
            explode=(0, 0.1),
            shadow=True,
            startangle=90)
axes[1].set_title('Class Distribution (Pie Chart)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Class distribution plot saved as '01_class_distribution.png'")

# 1.2 Correlation Heatmap
print("\n--- 1.2 Correlation Heatmap ---")
# Select a subset of features for better visualization
features_for_corr = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                      'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
                      'Time', 'Amount', 'Class']

plt.figure(figsize=(16, 14))
correlation_matrix = df[features_for_corr].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', 
            center=0, square=True, linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Correlation heatmap saved as '02_correlation_heatmap.png'")

# Show top correlations with Class
print("\nTop correlations with Class:")
class_corr = correlation_matrix['Class'].drop('Class').sort_values(key=abs, ascending=False)
print(class_corr.head(10))

# 1.3 Amount and Time Distribution by Class
print("\n--- 1.3 Amount and Time Distribution by Class ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Amount distribution - Histogram
axes[0, 0].hist(df[df['Class'] == 0]['Amount'], bins=50, alpha=0.7, 
                label='Legitimate', color='#2ecc71', edgecolor='black')
axes[0, 0].hist(df[df['Class'] == 1]['Amount'], bins=50, alpha=0.7, 
                label='Fraudulent', color='#e74c3c', edgecolor='black')
axes[0, 0].set_title('Amount Distribution by Class', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Amount')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 500)

# Amount distribution - Box plot
df.boxplot(column='Amount', by='Class', ax=axes[0, 1])
axes[0, 1].set_title('Amount Distribution by Class (Box Plot)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Class')
axes[0, 1].set_ylabel('Amount')
plt.suptitle('')

# Time distribution - Histogram
axes[1, 0].hist(df[df['Class'] == 0]['Time'], bins=50, alpha=0.7, 
                label='Legitimate', color='#2ecc71', edgecolor='black')
axes[1, 0].hist(df[df['Class'] == 1]['Time'], bins=50, alpha=0.7, 
                label='Fraudulent', color='#e74c3c', edgecolor='black')
axes[1, 0].set_title('Time Distribution by Class', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Time (seconds)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()

# Time distribution - Box plot
df.boxplot(column='Time', by='Class', ax=axes[1, 1])
axes[1, 1].set_title('Time Distribution by Class (Box Plot)', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Class')
axes[1, 1].set_ylabel('Time (seconds)')
plt.suptitle('')

plt.tight_layout()
plt.savefig('03_amount_time_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Amount and Time distribution plots saved as '03_amount_time_distribution.png'")

# Statistics by class
print("\nAmount statistics by class:")
print(df.groupby('Class')['Amount'].describe())
print("\nTime statistics by class:")
print(df.groupby('Class')['Time'].describe())

print("\n✅ Step 1 (EDA) completed!")
print("="*80)
print("Please confirm to proceed to Step 2 (Preprocessing)")
print("="*80)
