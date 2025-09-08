import numpy as np
import pandas as pd
from dset import df_no_NaN
from sklearn.model_selection import train_test_split

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

    return {
        "Accuracy": acc,
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
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Misclassification Rate: {misclass_rate:.3f}")
    print(f"True Positive Rate (Recall): {TPR:.3f}")
    print(f"False Positive Rate: {FPR:.3f}")
    print(f"True Negative Rate (Specificity): {TNR:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Prevalence: {prevalence:.3f}")
    
    return preds


def run_logistic_regression(X, y, score_cols, epochs=300, alfa=0.01):
    results = {}

    X = score_cols.values #score columns are dfx 
    y = target_col.values #and target columns are dfy

    # 2. Normalización (Z-score)
    X_mean = np.mean(X, axis=0)
    X_std  = np.std(X, axis=0)
    X_norm = (X - X_mean) / (X_std + 1e-9)

    # añadir columna de bias
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]

    for col in score_cols:
        print(f"\n=== Entrenando para {col} ===")

        # entrenar
        params = entrenar(X_bias, y, epochs=epochs, alfa=alfa)

        # evaluar
        preds, acc = evaluar(params, X_bias, y)

        print(f"Accuracy en {col}: {acc:.4f}")
        results[col] = {
            "params": params,
            "preds": preds,
            "accuracy": acc
        }


    return results

# Loop principal
score_cols = ["PTAT","STA","STR","DFM","RUA","RLS","RTP",
              "FTL","RW","RLR","FTA","FUA","RUH","RUW",
              "UCL","UDP","FTP"]

results = {}

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

    # Añadir columna de bias
    X_train_bias = np.c_[np.ones((X_train_norm.shape[0], 1)), X_train_norm]
    X_val_bias   = np.c_[np.ones((X_val_norm.shape[0], 1)), X_val_norm]

    # entrenar
    params = entrenar(X_bias, y, epochs=300, alfa=0.05)

    paramsChido = entrenar(X_train_bias, y_train, X_val_bias, y_val, epochs=300, alfa=0.05)

    # evaluar
    preds = evaluar(params, X_bias, y)
    acc = np.mean(preds==y)

    # metrics
    metrics = classification_metrics(y, preds)
    results[target_col] = metrics

    # imprimir métricas
    print(f"\n=== Results for {target_col} ===")
    for k,v in metrics.items():
        print(f"{k}: {v:.3f}" if isinstance(v,float) else f"{k}: {v}")


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


import matplotlib.pyplot as plt  #use this to generate a graph of the errors/loss so we can see whats going on (diagnostics)
'''
plt.plot(__errores__, label="Train cost")
plt.plot(certezaF2, label="Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Learning curve")
plt.title("Training loss")
plt.show()
'''


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
 