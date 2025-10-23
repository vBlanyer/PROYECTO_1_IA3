import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

# La funcion lo que hace es tomar de una vez el conjunto test generado en el preprocess_data del main.py para realizar las predicciones
def predict_from_csv(model, test_X, test_Y, test_size, apply_log=False):
      print("\n--- Predicción desde CSV ---")
      model.eval() # Pone el modelo en modo evaluación (eval()).

      with torch.no_grad(): # Desactiva el cálculo de gradientes para eficiencia
            y_pred_real = model(test_X) # salida cruda del modelo (logits, no probabilidades)
            probs = torch.sigmoid(y_pred_real)  # convierte logits a probabilidades entre 0 y 1 (No hacer esto si no estoy usando sigmoide)

            y_test_real = test_Y

            # En caso de no usar sigmoide o BCE
            '''if apply_log:
                  y_pred_real = np.expm1(y_pred_real.cpu().detach().numpy()).flatten() # Revertir la transformación logarítmica y aplicar flatten
                  y_test_real = np.expm1(y_test_real.cpu().detach().numpy()).flatten() # Revertir la transformación logarítmica y aplicar flatten
            else:
                  y_pred_real = y_pred_real.cpu().detach().numpy().flatten()
                  y_test_real = y_test_real.cpu().detach().numpy().flatten()'''
            
            '''if apply_log:
                  y_pred_real = np.expm1(probs.cpu().detach().numpy()).flatten() # Revertir la transformación logarítmica y aplicar flatten
                  y_test_real = np.expm1(y_test_real.cpu().detach().numpy()).flatten() # Revertir la transformación logarítmica y aplicar flatten
            else:'''
            y_pred_real = probs.cpu().detach().numpy().flatten()
            y_test_real = y_test_real.cpu().detach().numpy().flatten()

      # Dado que el modelo a veces predice una popularidad negativa (cuando la popularidad era de 0 a 100), esta mal,
      #  asi que seteamos estos casos a 0                        
      ''''for i in range(len(y_pred_real)):
            if y_pred_real[i] < 0:
                  y_pred_real[i] = 0'''
      
      print("len del conjunto de prueba:",len(test_X))

      '''rmse = np.sqrt(np.mean((y_pred_real - y_test_real) ** 2)) # Cálculo del RMSE (Root Mean Squared Error)
      print(f"RMSE en streams reales: {rmse:,.0f}")'''

      # Binarización de las predicciones para métricas de clasificación
      y_pred_binary = (y_pred_real >= 0.7).astype(int)

      # Métricas de clasificación binaria
      accuracy = accuracy_score(y_test_real, y_pred_binary)
      precision = precision_score(y_test_real, y_pred_binary)
      recall = recall_score(y_test_real, y_pred_binary)

      print(f"\nMétricas de clasificación binaria:")
      print(f"Accuracy:  {accuracy:.4f}")
      print(f"Precision: {precision:.4f}")
      print(f"Recall:    {recall:.4f}")

      print("\nEjemplos de predicción:")   
      indices = random.sample(range(len(y_pred_binary)), k=min(5, len(y_pred_binary)))
      for i in indices:
            print(f"Real: {int(y_test_real[i]):,} | Predicho: {int(y_pred_binary[i]):,}")