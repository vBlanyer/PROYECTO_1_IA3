import os
import time
import argparse
import logging
import pandas as pd
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Credenciales Spotify
client_id = "f9ad838952f244e4a431266da40515df"
client_secret = "6abc3e9f8c5349ef8b47514dfd491b83"
sp = Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))


def fetch_track_info_batch(track_ids):
    """Obtiene release_date y popularity de hasta 50 tracks por ID."""
    results = sp.tracks(track_ids)
    info = {}
    for track in results.get("tracks", []):
        if not track:
            continue
        tid = track["id"]
        release_date_full = track.get("album", {}).get("release_date")

        if not release_date_full:
            continue  # Si no hay fecha, descarta el track

        release_year = release_date_full[:4]

        popularity = track.get("popularity")

        # Validación: debe ser un número entre 0 y 100 la popularidad
        if popularity is None or not (0 <= popularity <= 100):
            print("track invalido en popularidad")
            continue    
        info[tid] = {"release_date": release_year, "popularity": popularity}

    return info


def process_file(input_path, output_path, query_col="track_id", encoding="utf-8", save_every=500):
    df = pd.read_csv(input_path, encoding=encoding)
    if query_col not in df.columns:
        raise ValueError(f"El CSV debe tener una columna '{query_col}'.")

    # 🧹 Limpiar IDs (extrae los 22 caracteres válidos)
    df[query_col] = df[query_col].astype(str).str.extract(r'([A-Za-z0-9]{22})')[0]
    df = df.dropna(subset=[query_col])

    # 🔧 Añadir columnas si no existen
    if "release_date" not in df.columns:
        df["release_date"] = pd.NA
    if "popularity" not in df.columns:
        df["popularity"] = pd.NA

    # Procesar TODAS las filas (sin filtro de NaN)
    pending = df[query_col].dropna().astype(str).unique().tolist()

    total = len(pending)
    logging.info("Tracks totales a procesar: %d", total)

    with tqdm(total=total, desc="Actualizando tracks", unit="track") as pbar:
        for i in range(0, total, 50):
            batch = pending[i:i + 50]
            try:
                batch_info = fetch_track_info_batch(batch)
            except Exception as e:
                logging.warning("Error en lote %d-%d: %s", i, i + 50, str(e))
                time.sleep(2)
                continue

            #  Reportar tracks no encontrados
            if len(batch_info) < len(batch):
                missing = set(batch) - set(batch_info.keys())
                if missing:
                    logging.warning("Tracks no encontrados (%d): %s", len(missing), list(missing)[:5])

            # Actualizar DataFrame
            for tid, info in batch_info.items():
                df.loc[df[query_col] == tid, ["release_date", "popularity"]] = [
                    info.get("release_date"),
                    info.get("popularity"),
                ]

            pbar.update(len(batch))

            # Guardado parcial cada cierto número de lotes
            if (i // 50) % save_every == 0 and i > 0:
                temp_path = output_path + ".partial.csv"
                df.to_csv(temp_path, index=False)
                logging.info("Guardado parcial en %s (hasta fila %d)", temp_path, i + 50)

            time.sleep(0.25)  # pausa ligera para no exceder rate limit

    # Guardado final
    df.to_csv(output_path, index=False)
    logging.info("Guardado final en %s", output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Actualiza release_date y popularity usando Spotify API.")
    parser.add_argument("--input", "-i", default="../data/dataset_cleaned.csv", help="Ruta CSV de entrada.")
    parser.add_argument("--output", "-o", default="spotify_tracks_updated.csv", help="Ruta CSV de salida.")
    parser.add_argument("--col", "-c", default="track_id", help="Columna con IDs de Spotify.")
    parser.add_argument("--encoding", "-e", default="utf-8", help="Encoding del archivo CSV.")
    parser.add_argument("--save-every", "-s", type=int, default=20, help="Guardar cada N lotes (50*N filas).")
    args = parser.parse_args()

    process_file(args.input, args.output, query_col=args.col, encoding=args.encoding, save_every=args.save_every)


if __name__ == "__main__":
    main()
