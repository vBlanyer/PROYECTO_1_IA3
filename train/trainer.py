import torch
from torch.utils.data import DataLoader, TensorDataset

# Función para entrenar el modelo. Usa DataLoader para manejar batches y optimización.
# - model: instancia de la red (como FeedforwardNN).
# - X_train, y_train: tensores de entrenamiento.
# - criterion: función de pérdida (por ejemplo, nn.MSELoss()).
# - optimizer: algoritmo de optimización (como torch.optim.Adam).
# - epochs: número de veces que se recorre todo el dataset.
# - batch_size: tamaño de cada lote de datos.
# ESTE USA RMSE
def train_model(model, X_train, y_train, criterion, optimizer, epochs=20, batch_size=32):
      dataset = TensorDataset(X_train, y_train) # Combina X_train y y_train en un solo objeto.
      loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # El DataLoader divide el dataset en lotes de tamaño batch_size y los
                                                                        # mezcla aleatoriamente (shuffle=True) en cada época.
                                                                        
      rmse_history = [] # para llevar un registro del RMSE por epoca
      
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
            
            # ahora evaluamos el RMSE sobre todo el conjunto de entrenamiento
            model.eval()
            with torch.no_grad():
                  predictions = model(X_train)
                  mse = criterion(predictions, y_train).item()
                  rmse = mse ** 0.5
                  rmse_history.append(rmse)
            
      return rmse_history

# ESTE USA BCE
def train_model_bce(model, X_train, y_train, criterion, optimizer, epochs=20, batch_size=32):
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    bce_loss_history = []  # Historial de pérdida binaria

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, BCE Loss: {total_loss:.4f}")

        # Evaluación sobre todo el conjunto de entrenamiento
        model.eval()
        with torch.no_grad():
            predictions = model(X_train)
            loss_epoch = criterion(predictions, y_train).item()
            bce_loss_history.append(loss_epoch)

    return bce_loss_history