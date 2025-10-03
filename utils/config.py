# Configuraciones clave para el proyecto, como rutas de datos, tamaños de capas de la red, tasa de aprendizaje, etc.
config = {
      "data_path": "data/dataset.csv",
      "data_path_cleaned": "data/dataset_cleaned.csv",
      "input_size": 18,  # Esto no se esta usando, ya que el input se autoprocesa en el preprocessor_data dependiendo de las columnas que encuentre
      "hidden_size": 64,
      "output_size": 1,
      "batch_size": 32,
      "learning_rate": 0.001,
      "epochs": 300,
      "test_path": "data/dataset_cleaned.csv"
}
