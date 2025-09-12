import numpy as np
import pandas as pd
from dset import df_no_NaN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Adaptar el data frame para volverlo un problema binario (media=0.5)
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
    
    # Calculate accuracy score (standard sklearn accuracy)
    accuracy_score_val = accuracy_score(y_true, y_pred)
    
    # Calculate bias score (difference between TPR and TNR)
    bias_score = abs(tpr - tnr)
    
    # Calculate variance score (measure of prediction consistency)
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

__errores__= [];  #global variable to store the errors/loss for visualisation
__certeza__ = [];

#intento de graficar todo en una sola gráfica:
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

#listas para almacenar valores de predicción de un toro nuevo
trained_params = {}
normalization_params = {}  

#-------------------------------------------------------------------------------------------------------------------
# θ_j := θ_j - (α/m) * Σ_{i=1 to m} [ (h_θ(x_i) - y_i) * x_i ]
#-------------------------------------------------------------------------------------------------------------------

#Puede reemplazarse con ReLU
def sigmoid(acum):
    return 1.0 / (1.0 + np.exp(-acum))

#Evaluación de hθ​(x)=σ(θTx)=1+e−θTx1​ (versión logística en vez de lineal)
def funcion_h(params, sample):
  acum = 0
  for i in range(len(params)): 
    acum += params[i] * sample[i] #Theta[i]*x[i] cada parametro por su variable
  return sigmoid(acum);            #h_θ(x_i): valor predicho por el modelo (sigmoide)

def funcion_costo(params, samples, y):
  global __errores__

  cost= 0

  for i in range(len(samples)):
    hyp = funcion_h(params, samples [i]) #h_θ(x_i)
    cost += -y[i]*np.log(hyp + 1e-9) - (1-y[i])*np.log(1-hyp + 1e-9) #J(θ)=−m1​∑[yi​log(hθ​(xi​))+(1−yi​)log(1−hθ​(xi​))]
    #error = hyp - y[i]                   #(h_θ(x_i) - y_i)
    #cost += error**2                     #((h_θ(x_i) - y_i)^2

  mean_cost = cost / len(samples)         #(1/m) * Σ (h_θ(x_i) - y_i)^2  (MSE sin el 1/2 factor. [2len(samples)])

  __errores__.append(mean_cost) #se guarda en la lista yay
  return mean_cost

def funcion_accuracy(params, samples, y):
  preds = [1 if funcion_h(params, sample) >= 0.5 else 0 for sample in samples]
  preds = np.array(preds)
  global __certeza__

  accuracy = 0 

  for i in range(len(samples)):

    TP = np.sum((preds == 1) & (y == 1))
    TN = np.sum((preds == 0) & (y == 0))
    
    accuracy = (TP+TN) / len(y)

  __certeza__.append(accuracy) #se guarda en la lista yay
  return accuracy

#θj​:=θj​−α⋅1/m ​i∑​(hθ​(xi​)−yi​)xi,j​
def funcion_update(params, samples, y, alfa):
  params_nuevo = list(params) #with the new values for the parameters after 1 run of the sample set (2 dimensiones)
  #por eso antes habíamos hecho (m, n) = samples.shape(), aunque eran params (son los que cambian, no los datos)
  #y en vez de for i in (m) y for j in (n):
  
  for j in range(len(params)):
    acum = 0

    for i in range(len(samples)):
      pred = funcion_h(params,samples[i])
      error = pred - y[i]
      acum += error * samples[i][j]
        #acum += funcion_h((samples[i], params[i], b)- Y[i])*samples[i][j]

    #theta_new[j] = theta[j] - (alpha/m) * grad
    params_nuevo[j] = params[j] - alfa*(1/len(samples))*acum
  return params_nuevo

#sgd_logistic regression
def entrenar(samples, y, val_samples=None, val_y= None, epochs=300, alfa=0.01):
    global __errores__
    __errores__ = []

    global train_losses, val_losses, train_accuracies, val_accuracies 
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # inicializamos parámetros en cero
    params = np.zeros(len(samples[0]))  # un parámetro por cada columna (incluyendo bias)

    for epoch in range(epochs): #imprimir los errores 
        
        params = funcion_update(params, samples, y, alfa)
        cost = funcion_costo(params, samples, y)
        cert = funcion_accuracy(params, samples, y)

        train_losses.append(cost)
        train_accuracies.append(cert)

        if val_samples is not None and val_y is not None:
          val_cost = funcion_costo(params, val_samples, val_y)
          val_acc = funcion_accuracy(params, val_samples, val_y)

          val_losses.append(val_cost)
          val_accuracies.append(val_acc)

          if epoch % 50 == 0:  # cada 50 epochs revisamos el costo pa no llenar tanto la terminal
              print(f"Epoch {epoch}, Train Cost: {cost:.4f}, Train Accuracy: {cert}")

              if val_samples is not None:
                 print(f"Epoch {epoch}, Val Cost: {val_cost:.4f}, Val Acc: {val_acc:.4f}")
    
    return params

def evaluar(params, samples, y):
    preds = [1 if funcion_h(params, sample) >= 0.5 else 0 for sample in samples]
    preds = np.array(preds)

    TP = np.sum((preds == 1) & (y == 1))
    TN = np.sum((preds == 0) & (y == 0))
    FP = np.sum((preds == 1) & (y == 0))
    FN = np.sum((preds == 0) & (y == 1))
    
    accuracy = (TP+TN) / len(y)
    misclass_rate = 1 - accuracy
    TPR = TP / (TP+FN+1e-9)
    FPR = FP / (FP+TN+1e-9)
    TNR = TN / (TN+FP+1e-9)
    precision = TP / (TP+FP+1e-9)
    prevalence = (TP+FN) / len(y)
    
    # Calculate additional metrics
    accuracy_score_val = accuracy_score(y, preds)
    bias_score = abs(TPR - TNR)
    variance_score = np.var(preds) if len(preds) > 0 else 0
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Accuracy Score: {accuracy_score_val:.3f}")
    print(f"Bias Score: {bias_score:.3f}")
    print(f"Variance Score: {variance_score:.3f}")
    print(f"Misclassification Rate: {misclass_rate:.3f}")
    print(f"True Positive Rate (Recall): {TPR:.3f}")
    print(f"False Positive Rate: {FPR:.3f}")
    print(f"True Negative Rate (Specificity): {TNR:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Prevalence: {prevalence:.3f}")
    
    return preds, TP, TN, FP, FN

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

# Loop principal
score_cols = ["PTAT","STA","STR","DFM","RUA","RLS","RTP",
              "FTL","RW","RLR","FTA","FUA","RUH","RUW",
              "UCL","UDP","FTP"]

results = {}
all_metrics = {}
confusion_matrix_data = {}

for target_col in score_cols:
    print(f"\n=== Entrenando para {target_col} ===")

    # binary target: above/below median
    median_val = df_no_NaN[target_col].median()
    y = (df_no_NaN[target_col] > median_val).astype(int).values

    # features = all other columns + bias (and without target columns)
    feature_cols = [c for c in score_cols if c != target_col]
    X = df_no_NaN[feature_cols].values
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]

    # Dividir entre sets de entrenar y validar 
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42) #aleatoriedad de 42 para reproducibilidad

    # Normalización (Z-score)
    X_mean = np.mean(X, axis=0)
    X_std  = np.std(X, axis=0)
    X_train_norm = (X_train - X_mean) / (X_std + 1e-9)
    X_val_norm = (X_val - X_mean) / (X_std + 1e-9)

    # Guardar parametros normalizados para prediccion de toro nuevo
    normalization_params[target_col] = {'mean': X_mean, 'std': X_std}

    # Añadir columna de bias
    X_train_bias = np.c_[np.ones((X_train_norm.shape[0], 1)), X_train_norm]
    X_val_bias   = np.c_[np.ones((X_val_norm.shape[0], 1)), X_val_norm]

    # entrenar
    params = entrenar(X_bias, y, epochs=300, alfa=0.05)
    paramsChido = entrenar(X_train_bias, y_train, X_val_bias, y_val, epochs=300, alfa=0.05)

    #Guardar parametros entrenados para predicción de toro nuevo
    trained_params[target_col] = paramsChido

    # evaluar
    preds, TP, TN, FP, FN = evaluar(params, X_bias, y)
    acc = np.mean(preds==y)

    # metrics
    metrics = classification_metrics(y, preds)
    results[target_col] = metrics
    all_metrics[target_col] = metrics
    
    # Store confusion matrix data
    confusion_matrix_data[target_col] = {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
    }

    # imprimir métricas
    print(f"\n=== Results for {target_col} ===")
    for k,v in metrics.items():
        if isinstance(v,float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")

# Plot individual confusion matrices for each feature
plot_individual_confusion_matrices(results, score_cols)

# Plot overall confusion matrix
total_TP, total_TN, total_FP, total_FN = plot_overall_confusion_matrix(results, score_cols)

# Calculate overall bias and variance scores
overall_bias_score = np.mean([all_metrics[col]['Bias Score'] for col in score_cols])
overall_variance_score = np.mean([all_metrics[col]['Variance Score'] for col in score_cols])

print(f"\n{'='*60}")
print("OVERALL SUMMARY")
print(f"{'='*60}")
print(f"Total True Positives (TP): {total_TP}")
print(f"Total True Negatives (TN): {total_TN}")
print(f"Total False Positives (FP): {total_FP}")
print(f"Total False Negatives (FN): {total_FN}")
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

# Create histogram plot
plt.figure(figsize=(14, 8))
x_pos = np.arange(len(comparison_df))
width = 0.15

# Plot each metric as separate bars
for i, metric in enumerate(metrics_to_compare):
    plt.bar(x_pos + i * width, comparison_df[metric], width, label=metric, alpha=0.8)

plt.xlabel('Target Variables')
plt.ylabel('Score')
plt.title('Logistic Regression Performance Metrics Across Different Targets', fontsize=16, fontweight='bold')
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




print(f"\n{'='*60}")
print("EXAMPLE PREDICTION: FARMER INVENTS A BULL")
print(f"{'='*60}")

# Example: farmer invents a bull with "average" values
invented_bull = {
    "PTAT": 1.5,
    "STA": 0.8,
    "STR": -0.2,
    "DFM": 1.2,
    "RUA": 0.0,
    "RLS": 0.5,
    "RTP": -0.1,
    "FTL": 1.0,
    "RW": 0.9,
    "RLR": 1.1,
    "FTA": 0.4,
    "FUA": -0.3,
    "RUH": 0.7,
    "RUW": 0.6,
    "UCL": -0.2,
    "UDP": 1.0,
    "FTP": 0.5
}

# For each target feature, predict whether the invented bull is above/below median
print("\nPredictions for the invented bull across all features:")
print("-" * 60)

for target_col in score_cols:
    # Get the median value for this target
    median_val = df_no_NaN[target_col].median()
    
    # Prepare features (exclude the target column)
    feature_cols = [c for c in score_cols if c != target_col]
    
    # Create input vector for the invented bull
    X_new = np.array([invented_bull[col] for col in feature_cols])
    
    # Normalize using the stored parameters
    norm_params = normalization_params[target_col]
    X_new_norm = (X_new - norm_params['mean']) / (norm_params['std'] + 1e-9)
    
    # Add bias term
    X_new_bias = np.concatenate([[1], X_new_norm])
    
    # Get the trained parameters
    params = trained_params[target_col]
    
    # Make prediction
    prediction_prob = funcion_h(params, X_new_bias)
    prediction = 1 if prediction_prob >= 0.5 else 0
    
    status = "ABOVE" if prediction == 1 else "BELOW"
    
    print(f"{target_col}: Median = {median_val:.2f}, Prediction = {status} median (prob: {prediction_prob:.3f})")




#promediar partes para graficar (porque ahora nos daría 5100 valores por correr los 17 features 300 veces.)
certezaF = []
certezaN = []
j = 0

for i in __certeza__:
   certezaN.append(i)
   j+=1
   if j==300:
      certezaF.append(certezaN)
      certezaN = []
      j=0
      
certezaF2 = []

for i in range(0,300):
   suma = 0
   for p in range(0,17):
      suma+=certezaF[p][i]
   certezaF2.append(suma/17)

# Intento de grafica con todo en uno
fig, ax1 = plt.subplots(figsize=(10, 6))

# Losses
color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color=color)
ax1.plot(train_losses, label='Training Loss', color='red', linestyle='-')
ax1.plot(val_losses, label='Validation Loss', color='red', linestyle='--')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, 1)  # Adjust based on your data

# crear otro eje y para accuracy
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color)
ax2.plot(train_accuracies, label='Training Accuracy', color='blue', linestyle='-')
ax2.plot(val_accuracies, label='Validation Accuracy', color='blue', linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 1)  # Accuracy ranges from 0 to 1

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

plt.title('Training Progress')
plt.show()
 




