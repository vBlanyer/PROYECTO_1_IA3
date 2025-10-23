# Análisis de Popularidad de Spotify Songs

**Autores:** 
* Blanyer Vielma, carnet 16-11238
* Franco Murillo, carnet 16-10782
* Miguel Gordillo, carnet 18-10807
## Descripción general
Este proyecto busca **predecir la popularidad de canciones en Spotify** utilizando una **red neuronal feedforward** desarrollada en **PyTorch**. El sistema permite entrenar el modelo, realizar predicciones manuales y ejecutar predicciones masivas a partir de archivos CSV.

## Dependencias principales

* Python 3.7+
* torch 2.6.0+cu118
* numpy 2.3.4
* pandas 2.3.3
* scikit-learn 1.7.2
* matplotlib 3.10.7
* tqdm 4.67.1
A
## Estructura del proyecto

```
PROYECTO_1_IA3/
├── data/
│   └── preprocessor.py           # Limpieza y preparación de los datos
├── models/
│   └── feedforward.py            # Definición del modelo (capas, activaciones)
├── train/
│   └── trainer.py                # Lógica de entrenamiento, batching y validación
├── utils/
│   ├── config.py                 # Hiperparámetros y rutas
│   ├── predict_manual.py         # Predicción interactiva
│   ├── predict_from_csv.py       # Predicción masiva por CSV
│   ├── clean_lines.py            # Limpieza de líneas dañadas del CSV
│   └── plots.py                  # Gráficas de pérdida y comparación
├── saved_models/                 # Modelos entrenados (.pth)
├── main.py                       # Script principal (menú de ejecución)
└── requirements.txt              # Dependencias
```


## Parametros configurables

Los parámetros principales se encuentran en `utils/config.py`:

```python
data_path = data/dataset.csv # Ruta al dataset original
data_path_cleaned = data/dataset_cleaned.csv # Ruta al dataset limpio
input_size = 18 # Número de características de entrada
hidden_size = 64 # Neuronas en capas ocultas
output_size = 1 # Neuronas en capa de salida
batch_size = 32 # Tamaño de lote para entrenamiento
learning_rate = 0.001 # Tasa de aprendizaje
epochs = 300 # Número de épocas de entrenamiento
test_path = data/dataset_cleaned.csv # Ruta al dataset de prueba
```

## Ejecución

Desde CMD o terminal en la raíz del proyecto:

```bash
# Menú interactivo
python main.py
```

## Referencias

* Dataset: *Spotify Tracks Dataset* — Kaggle (https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset/data)
* Documentación de PyTorch: https://pytorch.org/docs/stable/index.html
