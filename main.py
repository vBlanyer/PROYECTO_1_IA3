import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.preprocessor import preprocess_data
from models.feedforward import FeedforwardNN
from train.trainer import train_model, train_model_bce
from utils.config import config
from utils.predict_from_csv import predict_from_csv
from utils.clean_lines import clean_csv_file
from utils.plots import covergence_rmse, convergence_bce, real_vs_predict
from sklearn.metrics import precision_score, recall_score

# Debugging configuración del modelo (imprime los parámetros clave)
print("Configuración del modelo:")
for key, value in config.items():
    print(f"  {key}: {value}")

# ===============================
# ENTRENAMIENTO (Usando las diferentes funciones y clases definidas en los otros archivos)
# ===============================
# Limpieza inicial del CSV (elimina comas dentro de comillas dobles, puntos y comas finales, y comillas en los extremos).

#clean_csv_file(config["data_path"], config["data_path_cleaned"]) 

X_train, y_train, X_test, y_test = preprocess_data(config["data_path_cleaned"]) # Esto solo se ejecuta una vez
input_size = X_train.shape[1]
model = FeedforwardNN(input_size, config["hidden_size"], config["output_size"])

#criterion = torch.nn.MSELoss() # Función de pérdida (MSE para regresión)
#criterion = torch.nn.BCELoss()  # Funcion de perdida para clasificacion binaria (cuando uso sigmoide)
# Otra forma, para penalizar los falsos positivos. Cuenta cuántos ejemplos hay de cada clase en el conjunto de entrenamiento:
num_pos = (y_train == 1).sum().item()
num_neg = (y_train == 0).sum().item()
pos_weight = torch.tensor([num_neg / num_pos])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) # Para usar esto, no debo usar la sigmoide en el output.

optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"]) # Optimizador Adam con la tasa de aprendizaje del config

# Entrenamos el modelo (rmse_history se usara para graficar la convergencia)
# rmse_history = train_model(model, X_train, y_train, criterion, optimizer, config["epochs"], config["batch_size"])
bce_history = train_model_bce(model, X_train, y_train, criterion, optimizer, config["epochs"], config["batch_size"])

with torch.no_grad():
    '''''predictions = model(X_test) # Estamos pasando datos de entrada (X_test) a través del modelo para obtener una predicción. Internamente,
                                # esto ejecuta el método forward que se definió en la clase FeedforwardNN.
    test_loss = criterion(predictions, y_test).item()
    print(f"\nMSE en test: {test_loss:.4f}")'''
    # VERSION PARA BCE
    predictions = model(X_test)

    probs = torch.sigmoid(predictions) # valores entre 0 y 1 (es necesario hacer esto
                                       # ya que al hacer predicciones, debemos pasar los logits a probabilidades)
    
    loss_epoch = criterion(predictions, y_test).item()
    bce_history.append(loss_epoch)

    # Cálculo de accuracy
    pred_labels = (probs >= 0.7).float()
    correct = (pred_labels == y_test).sum().item()
    accuracy = correct / y_test.size(0)
    print(f"Accuracy: {accuracy:.4f}")


# Guardar modelo
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "feedforward_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en: {model_path}")

# ===============================
# CARGAR MODELO ENTRENADO Y PROBAR
# ===============================
print("\nProbando el modelo guardado...")
loaded_model = FeedforwardNN(input_size, config["hidden_size"], config["output_size"]) # Reconstruye el modelo con la misma arquitectura.
loaded_model.load_state_dict(torch.load(model_path))  # Carga los pesos guardados.
loaded_model.eval()   # Pone el modelo en modo evaluación (eval()).

# Verifica que el modelo cargado funciona igual que el original
with torch.no_grad():
    '''loaded_predictions = loaded_model(X_test)
    loaded_test_loss = criterion(loaded_predictions, y_test).item()
    #print(f"MSE del modelo cargado en test: {loaded_test_loss:.4f}")'''
    loaded_predictions = loaded_model(X_test)
    
    probs = torch.sigmoid(loaded_predictions) # valores entre 0 y 1 (es necesario hacer ya que BCEWithLogitsLoss espera logits como entrada, no probabilidades.

    loss_epoch = criterion(loaded_predictions, y_test).item()
    bce_history.append(loss_epoch)

    # Cálculo de accuracy
    '''pred_labels = (probs >= 0.5).float()
    correct = (pred_labels == y_test).sum().item()
    accuracy = correct / y_test.size(0)
    print(f"Accuracy: {accuracy:.4f}")'''

print("Entrenamiento completado.")

# Algunas predicciones dan menor a 0 (lo cual esta muy mal), asi que las seteamos a 0. Esto
# era para el caso de popularidad de 0 a 100
'''for i in range(len(loaded_predictions)):
    if loaded_predictions[i] < 0:
        loaded_predictions[i] = 0'''

# ===============================
# SELECCIÓN DEL MODO
# ===============================
print("\nSeleccion del modo de prueba:")
print("Opción: Archivo CSV (conjunto de prueba)")
print("Probando el modelo...")

csv_path = config["test_path"]  # Igual que el de config["data_path_cleaned"]
test_size = 0.2
#test_size = float(input("Porcentaje para test (ej: 0.2): ")) # Esto no se esta usando. Por defecto se esta usando 0.2
predict_from_csv(loaded_model, X_test, y_test, test_size)
    
# Graficas
#covergence_rmse(rmse_history)
convergence_bce(bce_history) # Para cuando uso clasificacion binaria para la popularidad
#real_vs_predict(y_test, loaded_predictions)
