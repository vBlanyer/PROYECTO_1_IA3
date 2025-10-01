import torch.nn as nn

# Clase que hereda de nn.Module, lo que la convierte en un modelo de PyTorch. Dentro de ella se define la arquitectura y el flujo de datos
class FeedforwardNN(nn.Module):
      # Constructor que define las capas de la red. input_size es el número de características de entrada (por ejemplo, 10 si se usan las columnas del CSV).
      # hidden_size es el número de neuronas en la capa oculta (un hiperparámetro que se puede ajustar).
      # output_size es el número de salidas (1 para regresión, más para clasificación multiclase).
      def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size) # Capa totalmente conectada de entrada a oculta
            self.relu = nn.ReLU()                          # Función de activación ReLU para introducir no linealidad
            self.fc2 = nn.Linear(hidden_size, output_size) # Capa totalmente conectada de oculta a salida

      # Este método define cómo fluye la información:
      # - x entra a la capa fc1.
      # - Se aplica ReLU para introducir no linealidad lo que permite que la red aprenda curvas, interacciones complejas entre variables y patrones
      # que no se pueden representar con líneas rectas.
      # - El resultado pasa a fc2, que genera la salida final.
      def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)