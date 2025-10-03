import torch
import os
import numpy as np
import pandas as pd
from data.preprocessor import preprocess_data
from models.feedforward import FeedforwardNN
from train.trainer import train_model
from utils.config import config
from utils.predict_manual import predict_manual
from utils.predict_from_csv import predict_from_csv
from utils.clean_lines import clean_csv_file
# Debugging configuración del modelo (imprime los parámetros clave)
print("Configuración del modelo:")
for key, value in config.items():
    print(f"  {key}: {value}")

# ===============================
# ENTRENAMIENTO (Usando las diferentes funciones y clases definidas en los otros archivos)
# ===============================
# Limpieza inicial del CSV (elimina comas dentro de comillas dobles, puntos y comas finales, y comillas en los extremos). Se realiza esto en el main 
# y no dentro de la funcion preprocess_data, para evitar hacer esto dos veces, esto es, dentro del siguiente preprocess_data, y el preprocess_data que 
# esta al final dentro de la funcion predict_from_csv, para que de esta forma optimizar y ademas evitar que el conjunto de test pueda cambiar al volver
# ejecutar: X_train, X_test, y_train, y_test = train_test_split(   
#               X_scaled, y, test_size=0.2, random_state=42      
#           )
clean_csv_file(config["data_path"], config["data_path_cleaned"]) 

X_train, y_train, X_test, y_test = preprocess_data(config["data_path_cleaned"]) # Esto solo se ejecuta una vez, ya que el de predict_from_csv lo comente
                                                                                # para que los encoders de los valores strings no cambien en la segunda llamada
input_size = X_train.shape[1]
model = FeedforwardNN(input_size, config["hidden_size"], config["output_size"])

criterion = torch.nn.MSELoss() # Función de pérdida (MSE para regresión)
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"]) # Optimizador Adam con la tasa de aprendizaje del config

train_model(model, X_train, y_train, criterion, optimizer, # Entrena el modelo
            config["epochs"], config["batch_size"])

with torch.no_grad():
    predictions = model(X_test) # Estamos pasando datos de entrada (X_test) a través del modelo para obtener una predicción. Internamente,
                                # esto ejecuta el método forward que se definió en la clase FeedforwardNN.
    test_loss = criterion(predictions, y_test).item()
    print(f"\nMSE en test: {test_loss:.4f}")

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
    loaded_predictions = loaded_model(X_test)
    loaded_test_loss = criterion(loaded_predictions, y_test).item()
    print(f"MSE del modelo cargado en test: {loaded_test_loss:.4f}")
print("Entrenamiento completado.")

# ===============================
# SELECCIÓN DEL MODO
# ===============================
print("\nSelecciona modo de prueba:")
print("1. Input manual")
print("2. Archivo CSV")
mode = input("Opción (1/2): ")

if mode == "1":
    predict_manual(loaded_model, input_size)
elif mode == "2":
    csv_path = config["test_path"]  # Igual que el de config["data_path_cleaned"]
    test_size = float(input("Porcentaje para test (ej: 0.2): ")) # Esto no se esta usando. Por defecto se esta usando 0.2
    # predict_from_csv(loaded_model, csv_path, test_size)   # Esta forma era la anterior, la cual cargaba el modelo limpio pero no procesado y por tanto
                                                            # se hacia otro preprocess_data dentro de la funcion, lo que puede hacer que los encoders 
                                                            # cambien y por ende, el ramdom=42 no se cumpla (la semilla para los conjuntos de entrenamiento
                                                            # y prueba cambien)
    predict_from_csv(loaded_model, X_test, y_test, test_size)                                                         
