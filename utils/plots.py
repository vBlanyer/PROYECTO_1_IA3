import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def covergence_rmse(rmse_history):
    """Grafica de la convergencia del RMSE (sobre el conjunto de entrenamiento)"""

    plt.figure(figsize=(8, 5))
    plt.plot(rmse_history, marker='o', linestyle='-', color='green')
    plt.xlabel('Época')
    plt.ylabel('RMSE')
    plt.title('Convergencia del RMSE durante el entrenamiento')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def convergence_bce(bce_loss_history):
    """Gráfica de la convergencia de la pérdida binaria (BCELoss) durante el entrenamiento"""

    plt.figure(figsize=(8, 5))
    plt.plot(bce_loss_history, marker='o', linestyle='-', color='purple')
    plt.xlabel('Época')
    plt.ylabel('BCELoss')
    plt.title('Convergencia de la pérdida binaria durante el entrenamiento')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
  
def real_vs_predict(y_test, loaded_predictions):
    """Grafica de los valores reales del conjunto de prueba vs los valores predichos por el modelo"""
    
    # Primero convertimos los tensores a arrays para poder graficar
    y_true = y_test.cpu().numpy().flatten()
    y_pred = loaded_predictions.cpu().numpy().flatten()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue', label='Predicciones')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal (y = x)')
    plt.xlabel('Valor real (popularidad)')
    plt.ylabel('Valor predicho')
    plt.title('Comparación entre valores reales y predichos')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
