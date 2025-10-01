import os
import torch
import numpy as np
from data.preprocessor import get_feature_names, get_label_encoders, get_scaler   

# Función que permite ingresar manualmente los valores de las características para hacer una predicción con el modelo entrenado.
def predict_manual(model, feature_count):
      print("\n--- Predicción manual ---")
      feature_names = get_feature_names()
      encoders = get_label_encoders()
      sample = []
      for name in feature_names:
            if name in encoders:
                  val_str = input(f"Introduce valor para '{name}' (nombre real): ")
                  encoder = encoders[name]
                  if val_str in encoder.classes_:
                        val = float(encoder.transform([val_str])[0])
                  else:
                        print(f"Error: '{val_str}' no se encuentra en el modelo para '{name}'.")
                        return
            else:
                  val = float(input(f"Introduce valor para '{name}': "))
            sample.append(val)
            
      scaler = get_scaler()
      sample_scaled = scaler.transform([sample])
      X_manual = torch.tensor(sample_scaled, dtype=torch.float32)
      with torch.no_grad():
            y_pred_log = model(X_manual)
            y_pred_real = np.expm1(y_pred_log.numpy())[0][0]

      print(f"\nPredicción de streams: {int(y_pred_real):,}")


