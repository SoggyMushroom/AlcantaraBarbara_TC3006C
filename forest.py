import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from dset import df_no_NaN

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def run_random_forest_analysis(df, score_cols, test_size=0.2, random_state=42):
    """
    Compact Random Forest analysis for multiple target variables
    """
    results = {}
    
    # Store all confusion matrices for later plotting
    all_confusion_matrices = {}
    
    for target_col in score_cols:
        print(f"\n=== Training Random Forest for {target_col} ===")
        
        # Create binary target: above/below median
        median_val = df[target_col].median()
        y = (df[target_col] > median_val).astype(int).values
        
        # Features: all other columns except target
        feature_cols = [c for c in score_cols if c != target_col]
        X = df[feature_cols].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1
        )
        
        rf.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = rf.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Store confusion matrix for combined plotting
        all_confusion_matrices[target_col] = cm
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store results
        results[target_col] = {
            'model': rf,
            'scaler': scaler,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'true_labels': y_test
        }
        
        # Print results (without individual plots)
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nTop 5 Important Features:")
        print(feature_importance.head())
    
    return results, all_confusion_matrices

def plot_all_confusion_matrices(confusion_matrices, score_cols):
    """
    Plot all confusion matrices in a single figure grid
    """
    n_cols = 4  # Number of columns in the grid
    n_rows = (len(score_cols) + n_cols - 1) // n_cols  # Calculate rows needed
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    
    # Calculate global min and max for consistent color scaling
    all_values = [cm.ravel() for cm in confusion_matrices.values()]
    global_min = min([np.min(values) for values in all_values])
    global_max = max([np.max(values) for values in all_values])
    
    for i, target_col in enumerate(score_cols):
        cm = confusion_matrices[target_col]
        
        # Create heatmap
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   cbar=False,  # Individual color bars
                   ax=axes[i],
                   vmin=global_min,
                   vmax=global_max,
                   square=True)
        
        axes[i].set_title(f'{target_col}\n(TN:{cm[0,0]} FP:{cm[0,1]}\nFN:{cm[1,0]} TP:{cm[1,1]})', 
                         fontsize=10, pad=10)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        axes[i].set_xticklabels(['Below', 'Above'])
        axes[i].set_yticklabels(['Below', 'Above'])
    
    # Hide empty subplots
    for i in range(len(score_cols), len(axes)):
        axes[i].set_visible(False)
    
    # Add a common colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap='Blues', 
                              norm=plt.Normalize(vmin=global_min, vmax=global_max))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    cbar_ax.set_ylabel('Count', rotation=270, labelpad=15)
    
    plt.suptitle('Confusion Matrices for All Target Variables', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])  # Make room for colorbar
    plt.show()

def plot_normalized_confusion_matrices(confusion_matrices, score_cols):
    """
    Plot normalized confusion matrices (percentages) in a single figure
    """
    n_cols = 4
    n_rows = (len(score_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()
    
    for i, target_col in enumerate(score_cols):
        cm = confusion_matrices[target_col]
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, 
                   annot=True, 
                   fmt='.2%', 
                   cmap='YlOrRd',
                   cbar=False,
                   ax=axes[i],
                   vmin=0,
                   vmax=1,
                   square=True)
        
        axes[i].set_title(f'{target_col}\nNormalized', fontsize=10, pad=10)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        axes[i].set_xticklabels(['Below', 'Above'])
        axes[i].set_yticklabels(['Below', 'Above'])
    
    # Hide empty subplots
    for i in range(len(score_cols), len(axes)):
        axes[i].set_visible(False)
    
    # Add common colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    cbar_ax.set_ylabel('Percentage', rotation=270, labelpad=15)
    
    plt.suptitle('Normalized Confusion Matrices (Percentages)', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.show()

# Main execution
if __name__ == "__main__":
    # Define your score columns
    score_cols = ["PTAT","STA","STR","DFM","RUA","RLS","RTP",
                  "FTL","RW","RLR","FTA","FUA","RUH","RUW",
                  "UCL","UDP","FTP"]
    
    # Run the analysis
    results, all_confusion_matrices = run_random_forest_analysis(df_no_NaN, score_cols)
    
    # Plot all confusion matrices together
    plot_all_confusion_matrices(all_confusion_matrices, score_cols)
    
    # Also plot normalized versions
    plot_normalized_confusion_matrices(all_confusion_matrices, score_cols)
    
    # Create other summary plots (feature importance, performance summary)
    # ... [include your other plotting functions here] ...
    
    # Print overall summary
    accuracies = [results[col]['accuracy'] for col in score_cols]
    print("\n" + "="*50)
    print("OVERALL SUMMARY")
    print("="*50)
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
    print(f"Accuracy Range: {min(accuracies):.4f} - {max(accuracies):.4f}")