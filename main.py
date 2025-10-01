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

# Debugging configuración
print("Configuración del modelo:")
for key, value in config.items():
    print(f"  {key}: {value}")

# ===============================
# ENTRENAMIENTO
# ===============================
X_train, y_train, X_test, y_test = preprocess_data(config["data_path"])
input_size = X_train.shape[1]
model = FeedforwardNN(input_size, config["hidden_size"], config["output_size"])

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

train_model(model, X_train, y_train, criterion, optimizer,
            config["epochs"], config["batch_size"])

with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test).item()
    print(f"\nMSE en test: {test_loss:.4f}")

# Guardar modelo
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)
model_path = os.path.join(save_dir, "feedforward_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Modelo guardado en: {model_path}")

# ===============================
# CARGAR MODELO
# ===============================
print("\nProbando el modelo guardado...")
loaded_model = FeedforwardNN(input_size, config["hidden_size"], 1)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()
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
    csv_path = config["test_path"]
    test_size = float(input("Porcentaje para test (ej: 0.2): "))
    predict_from_csv(loaded_model, csv_path, test_size)
