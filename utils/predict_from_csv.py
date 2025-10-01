import torch
import numpy as np
from data.preprocessor import preprocess_data

def predict_from_csv(model, csv_path, test_size=0.2):
      print("\n--- Predicción desde CSV ---")
      X_train, y_train, X_test, y_test = preprocess_data(csv_path)

      with torch.no_grad():
            y_pred_log = model(X_test)
            y_pred_real = np.expm1(y_pred_log.numpy()).flatten()
            y_test_real = np.expm1(y_test.numpy()).flatten()

      rmse = np.sqrt(np.mean((y_pred_real - y_test_real) ** 2))
      print(f"RMSE en streams reales: {rmse:,.0f}")

      print("\nEjemplos de predicción:")
      for i in range(min(5, len(y_pred_real))):
            print(f"Real: {int(y_test_real[i]):,} | Predicho: {int(y_pred_real[i]):,}")