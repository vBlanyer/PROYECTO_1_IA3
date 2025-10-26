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
│   └── dataset_cleaned.csv          # Conjunto de datos limpio inicial
│   └── preprocessor.py              # Limpieza y preparación de los datos
│   └── spotify_tracks_updated.csv    # Conjunto de datos actualizado (con release_date y popularidad mas reciente)
├── models/
│   └── feedforward.py            # Definición del modelo (capas, activaciones)
├── train/
│   └── trainer.py                # Lógica de entrenamiento, batching y validación
├── utils/
│   ├── add_data.py               # Conexion de API de Spotify, agrega atributo release_date al conjunto y actualiza la popullaridad
│   ├── config.py                 # Hiperparámetros y rutas
│   ├── correlacion.py            # Calcula la correlacion de los atributos con la popularidad
│   ├── predict_manual.py         # Predicción interactiva
│   ├── predict_from_csv.py       # Predicción masiva por CSV
│   ├── clean_lines.py            # Limpieza de líneas dañadas del CSV
│   └── plots.py                  # Gráficas de pérdida y comparación
├── saved_models/                 # Modelos entrenados (.pth)
├── main.py                       # Script principal (menú de ejecución)
```
## Estado Actual

Actualmente el modelo no realiza una limpieza de datos, ya que el conjunto ya fue limpiado. Se realiza es un preprocesamiento, como eliminar atributos innecesarios y normalizaciones.
Se trabaja es con el conjunto de datos spotify_tracks_updated.csv ya que es que tiene el atributo agregado release_date. 

La red tiene actualmente dos capas ocultas, con LeakyReLU como función de activacion entre las capas, y Sigmoide como función de activación para la capa output. 
Como función de pérdida se esta usando BCEWithLogitsLoss. 

El proceso actualmente realiza un preprocesamiento de los datos, entrenamiento de la red, guardado y carga del modelo entrenado y prueba del modelo con el conjunto de prueba, 
previamente extraido en la fase de preprocesamiento.


## Parametros configurables

Los parámetros principales se encuentran en `utils/config.py`:

```python
data_path = data/spotify_tracks_updated.csv # Ruta al dataset original (se usa el mismo ya que ya se realizó la limpieza)
data_path_cleaned = data/spotify_tracks_updated.csv # Ruta al dataset limpio
input_size = 18 # Número de características de entrada
hidden_size = 128 # Neuronas en capas ocultas
output_size = 1 # Neuronas en capa de salida
batch_size = 32 # Tamaño de lote para entrenamiento
learning_rate = 0.001 # Tasa de aprendizaje
epochs = 200 # Número de épocas de entrenamiento
test_path = data/spotify_tracks_updated.csv # Ruta al dataset de prueba
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
