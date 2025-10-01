_scaler = None

def get_scaler():
      return _scaler
_label_encoders = {}

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

def preprocess_data(path, target_column="streams"):
      df = pd.read_csv(path, encoding='ISO-8859-1')

      # Numero de columnnas a tomar en cuenta del cvs
      df = df.iloc[:, :9]

      # Debugging
      print("Previsualización de los datos extraídos del CSV:")
      print(df.head())

      # Aplicar función robusta de limpieza de streams
      df[target_column] = df[target_column].apply(clean_streams)
      
      # Transformación logarítmica del target (tomar en cuenta luego el proceso inverso para interpretación)
      df[target_column] = np.log1p(df[target_column])

      # Codificación de columnas de texto y guardar los encoders
      for col in ["track_name", "artist(s)_name"]:
            if col in df.columns:
                  encoder = LabelEncoder()
                  df[col] = encoder.fit_transform(df[col].astype(str))
                  _label_encoders[col] = encoder

      # Separación
      X = df.drop(columns=[target_column])
      y = df[target_column]

      # Escalado de features
      global _scaler
      _scaler = StandardScaler()
      X_scaled = _scaler.fit_transform(X)

      # División en train/test
      X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
      )

      # Tensores
      X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
      y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
      X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
      y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

      return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor