import re
import csv
from io import StringIO

def contar_atributos(linea: str) -> int:
    """
    Cuenta los atributos en una línea CSV respetando comillas dobles y comas internas.
    """
    buffer = StringIO(linea)
    lector = csv.reader(buffer, delimiter=',', quotechar='"')
    fila = next(lector)
    return len(fila)

def clean_csv_line(line):
    line = line.strip()

    # Paso 0: quitar los punto y coma finales
    for i in range(len(line)-1, -1, -1):
        if line[i] == ';':
            line = line[:i]
        else:
            break
    # Paso 1: quitar comillas dobles externas si existen
    if line[0] == '"' and line[len(line)-1] == '"':
        line = line[1:-1]


    # Paso 3: reemplazar comas internas dentro de campos entre comillas por espacios
    def replace_commas_inside_quotes(match):
        content = match.group(1)
        content_cleaned = content.replace(',', ' ')
        return f'"{content_cleaned}"'

    line = re.sub(r'"([^"]*?,[^"]*?)"', replace_commas_inside_quotes, line)

    if contar_atributos(line) < 21:
        return None  # ← Elimina la línea

    return line

def clean_csv_file(input_path, output_path):
    with open(input_path, "r", encoding="ISO-8859-1") as infile, open(output_path, "w", encoding="ISO-8859-1", newline='') as outfile:
        for line in infile:
            cleaned = clean_csv_line(line)
            if cleaned is not None:
                outfile.write(cleaned + "\n")
