_scaler = None # - el objeto StandardScaler que normaliza los datos numéricos.

_label_encoders = {} # - diccionario con los LabelEncoder usados para codificar texto

# Funciones que permiten acceder a los objetos globales desde otros módulos si se importa este archivo
def get_scaler():
      return _scaler
def get_label_encoders():
      return _label_encoders
def get_feature_names():
      columns = [
            "track_name",
            "artist(s)_name",
            "artist_count",
            "released_year",
            "released_month",
            "released_day",
            "in_spotify_playlists",
            "in_spotify_charts"
      ]
      return columns

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from utils.clean_streams import clean_streams

# Función que carga un CSV y lo transforma en tensores listos para entrenar un modelo.
def preprocess_data(path, target_column="streams"):
      df = pd.read_csv(path, encoding='ISO-8859-1')

      # Numero de columnnas a tomar en cuenta del cvs
      df = df.iloc[:, :9]

      # Debugging
      print("Previsualización de los datos extraídos del CSV:")
      print(df.head())

      # Aplicar función robusta de limpieza de streams (convierte a int y maneja errores)
      df[target_column] = df[target_column].apply(clean_streams)
      
      # Transformación logarítmica del target (tomar en cuenta luego el proceso inverso para interpretación)
      df[target_column] = np.log1p(df[target_column])

      # Codificación de columnas de texto a numeros para que el modelo pueda trabajar con ellos y guardar los encoders
      for col in ["track_name", "artist(s)_name"]:
            if col in df.columns:
                  encoder = LabelEncoder()
                  df[col] = encoder.fit_transform(df[col].astype(str))
                  _label_encoders[col] = encoder

      # Separación
      X = df.drop(columns=[target_column])
      y = df[target_column]

      # Escalado de features. Normaliza los datos para que tengan media 0 y desviación estándar 1.
      global _scaler
      _scaler = StandardScaler()
      X_scaled = _scaler.fit_transform(X)

      # Divide el conjunto en train/test
      X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
      )

      # Convierte los arrays en tensores de PyTorch, listos para entrenar una red neuronal.
      # Los tensores son la estructura base en PyTorch para cálculos, entrenamiento, y backpropagation.
      # Convertir los datos a float32 y darles la forma correcta evita errores de tipo y forma en el modelo.
      # El .view(-1, 1) es clave si se está entrenando una red con una sola salida.

      X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
      y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
      X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
      y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

      return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor # Datos que se usarán para entrenar y evaluar el modelo.