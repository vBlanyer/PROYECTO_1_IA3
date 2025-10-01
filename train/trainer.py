import torch
from torch.utils.data import DataLoader, TensorDataset

def train_model(model, X_train, y_train, criterion, optimizer, epochs=20, batch_size=32):
      dataset = TensorDataset(X_train, y_train)
      loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

      for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in loader:
                  outputs = model(inputs)
                  loss = criterion(outputs, targets)
                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()
                  total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
