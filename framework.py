import numpy as np
import pandas as pd
from dset import df_no_NaN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from xgboost.callback import TrainingCallback
import seaborn as sns


def classification_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    acc = (TP + TN) / len(y_true) if len(y_true) > 0 else 0
    mis_rate = 1 - acc
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0   # Recall
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0   # Specificity
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    prev = np.mean(y_true)
    
    accuracy_score_val = accuracy_score(y_true, y_pred)
    bias_score = abs(tpr - tnr)
    variance_score = np.var(y_pred) if len(y_pred) > 0 else 0

    return {
        "Accuracy": acc,
        "Accuracy Score": accuracy_score_val,
        "Bias Score": bias_score,
        "Variance Score": variance_score,
        "Misclassification Rate": mis_rate,
        "True Positive Rate (Recall)": tpr,
        "False Positive Rate": fpr,
        "True Negative Rate (Specificity)": tnr,
        "Precision": prec,
        "Prevalence": prev,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN
    }

# Global lists to store training history for all features
train_losses_all = []
val_losses_all = []
train_accuracies_all = []
val_accuracies_all = []
confusion_matrix_data = {}

class HistoryCallback(TrainingCallback):
    """Custom callback to track training history"""
    def __init__(self, X_train, y_train, X_val=None, y_val=None):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def custom_eval(self, y_true, y_pred_proba):
        """Custom evaluation function to track metrics"""
        y_pred = (y_pred_proba >= 0.5).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        logloss = -np.mean(y_true * np.log(y_pred_proba + 1e-9) + (1 - y_true) * np.log(1 - y_pred_proba + 1e-9))
        return accuracy, logloss
    
    def after_iteration(self, model, epoch, evals_log):
        """Called after each iteration"""
        # Get predictions
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        y_pred_train_proba = model.predict(dtrain)
        
        # Calculate training metrics
        train_acc, train_loss = self.custom_eval(self.y_train, y_pred_train_proba)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        
        # Calculate validation metrics if available
        if self.X_val is not None and self.y_val is not None:
            dval = xgb.DMatrix(self.X_val, label=self.y_val)
            y_pred_val_proba = model.predict(dval)
            val_acc, val_loss = self.custom_eval(self.y_val, y_pred_val_proba)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
        
        if epoch % 50 == 0:
            print(f"Iteration {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")
            if self.X_val is not None:
                print(f"              Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        return False  # Continue training

    def get_history(self):
        """Return the training history"""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }

def plot_individual_confusion_matrices(results, score_cols):
    """Plot confusion matrices for each feature in a single image"""
    n_features = len(score_cols)
    n_cols = 4  # Number of columns in the subplot grid
    n_rows = (n_features + n_cols - 1) // n_cols  # Calculate number of rows needed
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes for easy indexing
    
    for i, target_col in enumerate(score_cols):
        ax = axes[i]
        metrics = results[target_col]
        
        # Create confusion matrix data
        cm_data = np.array([[metrics['TN'], metrics['FP']],
                           [metrics['FN'], metrics['TP']]])
        
        # Plot heatmap
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'])
        
        ax.set_title(f'{target_col}\nTP: {metrics["TP"]}, TN: {metrics["TN"]}\nFP: {metrics["FP"]}, FN: {metrics["FN"]}', 
                    fontsize=10, pad=10)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Confusion Matrices for Each Feature', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()

def plot_overall_confusion_matrix(results, score_cols):
    """Plot overall confusion matrix across all targets"""
    total_TP = sum([results[col]['TP'] for col in score_cols])
    total_TN = sum([results[col]['TN'] for col in score_cols])
    total_FP = sum([results[col]['FP'] for col in score_cols])
    total_FN = sum([results[col]['FN'] for col in score_cols])
    
    # Create overall confusion matrix
    cm_data = np.array([[total_TN, total_FP],
                       [total_FN, total_TP]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    
    plt.title('Overall Confusion Matrix (All Features Combined)\n'
             f'Total TP: {total_TP}, Total TN: {total_TN}\n'
             f'Total FP: {total_FP}, Total FN: {total_FN}', 
             fontsize=14, pad=20)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    return total_TP, total_TN, total_FP, total_FN

def train_xgboost_with_history(X_train, y_train, X_val=None, y_val=None, epochs=300, learning_rate=0.05):
    """Train XGBoost with history tracking using proper callback"""
    
    # Create callback instance
    history_callback = HistoryCallback(X_train, y_train, X_val, y_val)
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Set up parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'error'],
        'max_depth': 3,
        'learning_rate': learning_rate,
        'subsample': 0.07,
        'colsample_bytree': 0.7,

        'reg_alpha': 0.5,  # L1 regularization
        'reg_lambda': 1.3,  # L2 regularization

        'random_state': 42

    }
    
    # Train with validation if provided
    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        model = xgb.train(
            params, 
            dtrain, 
            num_boost_round=epochs,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=50,
            callbacks=[history_callback]
        )
    else:
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=epochs,
            verbose_eval=50,
            callbacks=[history_callback]
        )
    
    return model, history_callback

def evaluate_xgboost(model, X, y):
    """Evaluate XGBoost model"""
    dtest = xgb.DMatrix(X)
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    metrics = classification_metrics(y, y_pred)
    
    print(f"Accuracy: {metrics['Accuracy']:.3f}")
    print(f"Accuracy Score: {metrics['Accuracy Score']:.3f}")
    print(f"Bias Score: {metrics['Bias Score']:.3f}")
    print(f"Variance Score: {metrics['Variance Score']:.3f}")
    print(f"Misclassification Rate: {metrics['Misclassification Rate']:.3f}")
    print(f"True Positive Rate (Recall): {metrics['True Positive Rate (Recall)']:.3f}")
    print(f"False Positive Rate: {metrics['False Positive Rate']:.3f}")
    print(f"True Negative Rate (Specificity): {metrics['True Negative Rate (Specificity)']:.3f}")
    print(f"Precision: {metrics['Precision']:.3f}")
    print(f"Prevalence: {metrics['Prevalence']:.3f}")
    
    return y_pred, metrics


# Add ROC curve plotting function
def plot_roc_curves_xgboost(models_dict, X_data, y_data, scalers, score_cols):
    """Plot ROC curves for all features and an overall ROC curve for XGBoost"""
    plt.figure(figsize=(12, 10))
    
    # Colors for individual ROC curves
    colors = plt.cm.tab20(np.linspace(0, 1, len(score_cols)))
    
    # Plot ROC curves for each feature
    for i, target_col in enumerate(score_cols):
        # Get the model and scaler for this feature
        model = models_dict[target_col]['model']
        scaler = scalers[target_col]
        
        # Scale the data
        X_scaled = scaler.transform(X_data[target_col])
        
        # Create DMatrix and get predictions
        dtest = xgb.DMatrix(X_scaled)
        y_pred_proba = model.predict(dtest)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_data[target_col], y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color=colors[i], lw=2, 
                label=f'{target_col} (AUC = {roc_auc:.3f})', alpha=0.7)
    
    # Plot overall ROC curve (average of all predictions)
    all_probs = []
    all_y = []
    for target_col in score_cols:
        model = models_dict[target_col]['model']
        scaler = scalers[target_col]
        X_scaled = scaler.transform(X_data[target_col])
        
        dtest = xgb.DMatrix(X_scaled)
        y_pred_proba = model.predict(dtest)
        all_probs.extend(y_pred_proba)
        all_y.extend(y_data[target_col])
    
    fpr_overall, tpr_overall, _ = roc_curve(all_y, all_probs)
    roc_auc_overall = auc(fpr_overall, tpr_overall)
    
    plt.plot(fpr_overall, tpr_overall, color='black', lw=3, 
            label=f'Overall (AUC = {roc_auc_overall:.3f})', linestyle='--')
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.5)')
    
    # Format the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('XGBoost ROC Curves for All Features and Overall Performance')
    plt.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig('xgboost_roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return roc_auc_overall


# Main execution
score_cols = ["PTAT","STA","STR","DFM","RUA","RLS","RTP",
              "FTL","RW","RLR","FTA","FUA","RUH","RUW",
              "UCL","UDP","FTP"]

results = {}
all_metrics = {}
confusion_matrix_data = {}
X_data_dict = {}  # Store X data for each target
y_data_dict = {}  # Store y data for each target
models_dict = {}  # Store models for each target
scalers_dict = {}  # Store scalers for each target

for target_col in score_cols:
    print(f"\n=== Training XGBoost for {target_col} ===")

    # Binary target: above/below median
    median_val = df_no_NaN[target_col].median()
    y = (df_no_NaN[target_col] > median_val).astype(int).values

    # Features = all other columns
    feature_cols = [c for c in score_cols if c != target_col]
    X = df_no_NaN[feature_cols].values

    # Store data for ROC curve calculation
    X_data_dict[target_col] = X
    y_data_dict[target_col] = y

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Store scaler for this target
    scalers_dict[target_col] = scaler

    # Train XGBoost with history tracking
    model, history_callback = train_xgboost_with_history(X_train_scaled, y_train, X_val_scaled, y_val, epochs=300, learning_rate=0.05)

    # Store model for this target
    models_dict[target_col] = {
        'model': model,
        'history': history_callback.get_history()
    }

    # Store training history for overall metrics
    history = history_callback.get_history()
    train_losses_all.append(history['train_losses'])
    val_losses_all.append(history['val_losses'])
    train_accuracies_all.append(history['train_accuracies'])
    val_accuracies_all.append(history['val_accuracies'])

    # Evaluate on full dataset
    X_full_scaled = scaler.transform(X)
    y_pred, metrics = evaluate_xgboost(model, X_full_scaled, y)

    # Store results
    results[target_col] = metrics
    all_metrics[target_col] = metrics
    confusion_matrix_data[target_col] = {
        'TP': metrics['TP'], 'TN': metrics['TN'], 
        'FP': metrics['FP'], 'FN': metrics['FN']
    }

    # Print metrics
    print(f"\n=== Results for {target_col} ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")

    # Store training history for overall metrics
    history = history_callback.get_history()
    train_losses_all.append(history['train_losses'])
    val_losses_all.append(history['val_losses'])
    train_accuracies_all.append(history['train_accuracies'])
    val_accuracies_all.append(history['val_accuracies'])

    # Evaluate on full dataset
    X_full_scaled = scaler.transform(X)
    y_pred, metrics = evaluate_xgboost(model, X_full_scaled, y)

    # Store results
    results[target_col] = metrics
    all_metrics[target_col] = metrics
    confusion_matrix_data[target_col] = {
        'TP': metrics['TP'], 'TN': metrics['TN'], 
        'FP': metrics['FP'], 'FN': metrics['FN']
    }

    # Print metrics
    print(f"\n=== Results for {target_col} ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")

# Plot confusion matrices
plot_individual_confusion_matrices(results, score_cols)
total_TP, total_TN, total_FP, total_FN = plot_overall_confusion_matrix(results, score_cols)

# Calculate overall bias and variance scores
overall_bias_score = np.mean([all_metrics[col]['Bias Score'] for col in score_cols])
overall_variance_score = np.mean([all_metrics[col]['Variance Score'] for col in score_cols])

print(f"\n{'='*60}")
print("OVERALL SUMMARY - XGBoost")
print(f"{'='*60}")
print(f"Total True Positives (TP): {total_TP}")
print(f"Total True Negatives (TN): {total_TN}")
print(f"Total False Positives (FP): {total_FP}")
print(f"Total False Negatives (FN): {total_FN}")
print(f"Overall Bias Score (Average): {overall_bias_score:.3f}")
print(f"Overall Variance Score (Average): {overall_variance_score:.3f}")

# Calculate overall training metrics
def calculate_overall_metrics(metrics_list):
    """Calculate average metrics across all features"""
    min_length = min(len(m) for m in metrics_list)
    overall_metrics = []
    
    for i in range(min_length):
        avg_value = np.mean([m[i] for m in metrics_list if i < len(m)])
        overall_metrics.append(avg_value)
    
    return overall_metrics

train_losses_avg = calculate_overall_metrics(train_losses_all)
val_losses_avg = calculate_overall_metrics(val_losses_all)
train_accuracies_avg = calculate_overall_metrics(train_accuracies_all)
val_accuracies_avg = calculate_overall_metrics(val_accuracies_all)

# Print overall training metrics
print(f"\nOverall Training Metrics:")
print(f"Final Training Loss: {train_losses_avg[-1]:.4f}")
print(f"Final Validation Loss: {val_losses_avg[-1]:.4f}")
print(f"Final Training Accuracy: {train_accuracies_avg[-1]:.4f}")
print(f"Final Validation Accuracy: {val_accuracies_avg[-1]:.4f}")

# Plot overall training progress
fig, ax1 = plt.subplots(figsize=(10, 6))

# Losses
color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color=color)
ax1.plot(train_losses_avg, label='Training Loss', color='red', linestyle='-')
ax1.plot(val_losses_avg, label='Validation Loss', color='red', linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, max(max(train_losses_avg), max(val_losses_avg)) * 1.1)

# Accuracy
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(train_accuracies_avg, label='Training Accuracy', color='blue', linestyle='-')
ax2.plot(val_accuracies_avg, label='Validation Accuracy', color='blue', linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

plt.title('Overall Training Progress (Average Across All Features)')
plt.tight_layout()
plt.show()

# Plot comparison of model performance across different targets
metrics_to_compare = ['Accuracy Score', 'Bias Score', 'Precision', 
                     'True Positive Rate (Recall)', 'True Negative Rate (Specificity)']

comparison_data = []
for target in score_cols:
    row = [target]
    for metric in metrics_to_compare:
        row.append(all_metrics[target][metric])
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data, columns=['Target'] + metrics_to_compare)
comparison_df = comparison_df.sort_values('Accuracy Score', ascending=False)

# Create histogram plot
plt.figure(figsize=(14, 8))
x_pos = np.arange(len(comparison_df))
width = 0.15

# Plot each metric as separate bars
for i, metric in enumerate(metrics_to_compare):
    plt.bar(x_pos + i * width, comparison_df[metric], width, label=metric, alpha=0.8)

plt.xlabel('Target Variables')
plt.ylabel('Score')
plt.title('XGBoost Performance Metrics Across Different Targets', fontsize=16, fontweight='bold')
plt.xticks(x_pos + width * 2, comparison_df['Target'], rotation=45, ha='right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Print summary of best performing models
print(f"\n{'='*60}")
print("Top 5 Best Performing Models by Accuracy Score")
print(f"{'='*60}")

top_5 = comparison_df.nlargest(5, 'Accuracy Score')[['Target', 'Accuracy Score', 'Bias Score', 'Precision']]
for i, (idx, row) in enumerate(top_5.iterrows(), 1):
    print(f"{i}. {row['Target']}: Accuracy={row['Accuracy Score']:.3f}, Bias={row['Bias Score']:.3f}, Precision={row['Precision']:.3f}")

# Print summary of least biased models
print(f"\n{'='*60}")
print("Top 5 Least Biased Models (Lowest Bias Score)")
print(f"{'='*60}")

least_biased = comparison_df.nsmallest(5, 'Bias Score')[['Target', 'Accuracy Score', 'Bias Score', 'Precision']]
for i, (idx, row) in enumerate(least_biased.iterrows(), 1):
    print(f"{i}. {row['Target']}: Accuracy={row['Accuracy Score']:.3f}, Bias={row['Bias Score']:.3f}, Precision={row['Precision']:.3f}")
 

# Add ROC curve plotting after all other plots
print("\nGenerating XGBoost ROC curves comparison...")
overall_auc = plot_roc_curves_xgboost(models_dict, X_data_dict, y_data_dict, scalers_dict, score_cols)
print(f"Overall AUC: {overall_auc:.3f}")