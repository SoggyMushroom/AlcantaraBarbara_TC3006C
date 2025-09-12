import numpy as np
import pandas as pd
from dset import df_no_NaN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from xgboost.callback import TrainingCallback

# Install xgboost if not already installed: pip install xgboost

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

#intento de graficar todo en una sola gráfica:
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

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

        global train_loss, train_acc, val_loss, val_acc
        
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

    def plot_training_progress(self, target_col):
        """Create the combined training progress plot"""

        train_losses.append(self.train_losses)
        train_accuracies.append(self.train_accuracies)
        val_losses.append(self.val_losses)
        val_accuracies.append(self.val_accuracies)

        if not self.train_losses:
            return
        
        epochs = range(len(self.train_losses))
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Losses on left y-axis
        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(epochs, self.train_losses, label='Training Loss', color='red', linestyle='-')
        if self.val_losses:
            ax1.plot(epochs, self.val_losses, label='Validation Loss', color='red', linestyle='--')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, max(max(self.train_losses), max(self.val_losses) if self.val_losses else max(self.train_losses)) * 1.1)
        
        # Accuracy on right y-axis
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(epochs, self.train_accuracies, label='Training Accuracy', color='blue', linestyle='-')
        if self.val_accuracies:
            ax2.plot(epochs, self.val_accuracies, label='Validation Accuracy', color='blue', linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1)  # Accuracy ranges from 0 to 1
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        plt.title(f'Training Progress - {target_col}')
        plt.tight_layout()
        plt.show()

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
        'max_depth': 6,
        'learning_rate': learning_rate,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
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

# Main execution
score_cols = ["PTAT","STA","STR","DFM","RUA","RLS","RTP",
              "FTL","RW","RLR","FTA","FUA","RUH","RUW",
              "UCL","UDP","FTP"]

results = {}
all_metrics = {}

for target_col in score_cols:
    print(f"\n=== Training XGBoost for {target_col} ===")

    # Binary target: above/below median
    median_val = df_no_NaN[target_col].median()
    y = (df_no_NaN[target_col] > median_val).astype(int).values

    # Features = all other columns
    feature_cols = [c for c in score_cols if c != target_col]
    X = df_no_NaN[feature_cols].values

    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train XGBoost with history tracking
    model, history_callback = train_xgboost_with_history(X_train_scaled, y_train, X_val_scaled, y_val, epochs=300, learning_rate=0.05)

    # Plot training progress
    history_callback.plot_training_progress(target_col)

    # Evaluate on full dataset
    X_full_scaled = scaler.transform(X)
    y_pred, metrics = evaluate_xgboost(model, X_full_scaled, y)

    # Store results
    results[target_col] = metrics
    all_metrics[target_col] = metrics

    # Print metrics
    print(f"\n=== Results for {target_col} ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")

# Calculate overall bias and variance scores
overall_bias_score = np.mean([all_metrics[col]['Bias Score'] for col in score_cols])
overall_variance_score = np.mean([all_metrics[col]['Variance Score'] for col in score_cols])

print(f"\n{'='*60}")
print("OVERALL SUMMARY - XGBoost")
print(f"{'='*60}")
print(f"Overall Bias Score (Average): {overall_bias_score:.3f}")
print(f"Overall Variance Score (Average): {overall_variance_score:.3f}")

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

print(comparison_df)

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

least_biased = comparison_df.nsmallest(5, 'Bias Score')[['Target', 'Accuracy Score', 'Bias Score']]
for i, (idx, row) in enumerate(least_biased.iterrows(), 1):
    print(f"{i}. {row['Target']}: Accuracy={row['Accuracy Score']:.3f}, Bias={row['Bias Score']:.3f}")


train_losses_g = []
val_losses_g = []
train_accuracies_g = []
val_accuracies_g = []

for i in range(0, len(train_losses[0])-1):

    suma_l = 0
    suma_vl = 0
    suma_ta = 0
    suma_va = 0

    for p in range(0,17):
        
        print(train_losses[p][i])
        suma_l += train_losses[p][i]
        suma_vl += val_losses[p][i]
        suma_ta += train_accuracies[p][i]
        suma_va += val_accuracies[p][i]

    train_losses_g.append(suma_l/17)
    val_losses_g.append(suma_vl/17)
    train_accuracies_g.append(suma_ta/17)
    val_accuracies_g.append(suma_va/17)


# Intento de grafica con todo en uno
fig, ax1 = plt.subplots(figsize=(10, 6))

# Losses
color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color=color)
ax1.plot(train_losses_g, label='Training Loss', color='red', linestyle='-')
ax1.plot(val_losses_g, label='Validation Loss', color='red', linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 1)  # Adjust based on your data

# crear otro eje y para accuracy
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(train_accuracies_g, label='Training Accuracy', color='blue', linestyle='-')
ax2.plot(val_accuracies_g, label='Validation Accuracy', color='blue', linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1)  # Accuracy ranges from 0 to 1

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

plt.title('Training Progress')
plt.show()
 