import torch
import numpy as np
from data.preprocessor import preprocess_data

# Función que toma un modelo entrenado y un CSV, y realiza predicciones mostrando resultados comparativos.
# Hace predicciones, revierte la transformación logarítmica aplicada durante el preprocesamiento, calcula el error (RMSE), y muestra
# ejemplos comparativos.
# def predict_from_csv(model, csv_path, test_size, apply_log=False):

# No hago la funcion anterior para no volver a ejecutar el preprocess_data y evitar que los econders cambien. Por tanto, la funcion lo que hace es tomar
# de una vez el conjunto test generado en el preprocess_data del main.py para realizar las predicciones
def predict_from_csv(model, test_X, test_Y, test_size, apply_log=False):
      print("\n--- Predicción desde CSV ---")
      model.eval() # Pone el modelo en modo evaluación (eval()).
      #X_train, y_train, X_test, y_test = preprocess_data(csv_path) # Preprocesamiento de datos. Lo que se hacia para el primer def de arriba.

      with torch.no_grad(): # Desactiva el cálculo de gradientes para eficiencia
            y_pred_real = model(test_X)
            y_test_real = test_Y

            if apply_log:
                  y_pred_real = np.expm1(y_pred_real.cpu().detach().numpy()).flatten() # Revertir la transformación logarítmica y aplicar flatten
                  y_test_real = np.expm1(y_test_real.cpu().detach().numpy()).flatten() # Revertir la transformación logarítmica y aplicar flatten
            else:
                  y_pred_real = y_pred_real.cpu().detach().numpy().flatten()
                  y_test_real = y_test_real.cpu().detach().numpy().flatten()


      rmse = np.sqrt(np.mean((y_pred_real - y_test_real) ** 2)) # Cálculo del RMSE (Root Mean Squared Error)
      print(f"RMSE en streams reales: {rmse:,.0f}")

      print("\nEjemplos de predicción:")
      for i in range(min(5, len(y_pred_real))):
            print(f"Real: {int(y_test_real[i]):,} | Predicho: {int(y_pred_real[i]):,}")