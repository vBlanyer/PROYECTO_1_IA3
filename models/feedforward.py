import torch.nn as nn

# Clase que hereda de nn.Module, lo que la convierte en un modelo de PyTorch. Dentro de ella se define la arquitectura y el flujo de datos
class FeedforwardNN(nn.Module):
      # Constructor que define las capas de la red. input_size es el número de características de entrada (por ejemplo, 10 si se usan las columnas del CSV).
      # hidden_size es el número de neuronas en la capa oculta (un hiperparámetro que se puede ajustar).
      # output_size es el número de salidas (1 para regresión, más para clasificación multiclase).
      def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            # Red de 1 capa:
            '''self.fc1 = nn.Linear(input_size, hidden_size) # Capa totalmente conectada de entrada a oculta
            #self.relu = nn.ReLU()                          # Función de activación ReLU para introducir no linealidad
            # Usaremos la funcion de activacion LeakyReLU en lugar de ReLU ya que ReLU convierte valores negativos a 0 despues de la primera capa, lo que
            # nos puede hacer perder informacion util. 
            self.relu = nn.LeakyReLU(0.01)                 # Función de activación LeakyReLU para introducir no linealidad
            self.fc2 = nn.Linear(hidden_size, output_size) # Capa totalmente conectada de oculta a salida'''

            # Red de 2 capas:
            self.fc1 = nn.Linear(input_size, hidden_size)       # Entrada → primera capa oculta
            self.relu1 = nn.LeakyReLU(0.01)                      # Activación 1
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2) # Segunda capa oculta (menos neuronas)
            self.relu2 = nn.LeakyReLU(0.01)                      # Activación 2
            self.fc3 = nn.Linear(hidden_size // 2, output_size) # Capa de salida



      # Este método define cómo fluye la información:
      # - x entra a la capa fc1.
      # - Se aplica ReLU para introducir no linealidad lo que permite que la red aprenda curvas, interacciones complejas entre variables y patrones
      # que no se pueden representar con líneas rectas.
      # - El resultado pasa a fc2, que genera la salida final.
      def forward(self, x):
            # Red de 1 capa:
            '''x = self.relu(self.fc1(x))
            return self.fc2(x)'''
      
            # Red de 2 capas:
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            return self.fc3(x)
