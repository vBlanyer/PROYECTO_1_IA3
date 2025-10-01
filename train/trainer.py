import torch
from torch.utils.data import DataLoader, TensorDataset

# Función para entrenar el modelo. Usa DataLoader para manejar batches y optimización.
# - model: instancia de la red (como FeedforwardNN).
# - X_train, y_train: tensores de entrenamiento.
# - criterion: función de pérdida (por ejemplo, nn.MSELoss()).
# - optimizer: algoritmo de optimización (como torch.optim.Adam).
# - epochs: número de veces que se recorre todo el dataset.
# - batch_size: tamaño de cada lote de datos.
def train_model(model, X_train, y_train, criterion, optimizer, epochs=20, batch_size=32):
      dataset = TensorDataset(X_train, y_train) # Combina X_train y y_train en un solo objeto.
      loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # El DataLoader divide el dataset en lotes de tamaño batch_size y los
                                                                        # mezcla aleatoriamente (shuffle=True) en cada época.
      # Bucle de entrenamiento
      for epoch in range(epochs):
            total_loss = 0
            # Itera sobre cada lote de datos
            for inputs, targets in loader:
                  outputs = model(inputs) # Forward pass: calcula las predicciones del modelo.
                  loss = criterion(outputs, targets) # Calcula la pérdida entre las predicciones y los valores reales.
                  optimizer.zero_grad() # Limpia los gradientes acumulados.
                  loss.backward() # Backward pass: calcula los gradientes.
                  optimizer.step() # Actualiza los pesos del modelo.
                  total_loss += loss.item() # Acumula la pérdida para monitorear el progreso.
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}") # Imprime la pérdida total al final de cada época.
