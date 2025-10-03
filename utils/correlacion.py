import pandas as pd
from sklearn.preprocessing import LabelEncoder
from itertools import combinations

def correlacion_individual(df: pd.DataFrame, target: str) -> pd.Series:
    """
    Calcula la correlación de cada columna con el target.
    Codifica columnas categóricas si es necesario.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            df_copy[col] = LabelEncoder().fit_transform(df_copy[col].astype(str))
    return df_copy.corr()[target].drop(target).sort_values(ascending=False)

def correlacion_doble(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Calcula la correlación entre el target y el producto de cada par de columnas.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            df_copy[col] = LabelEncoder().fit_transform(df_copy[col].astype(str))

    resultados = []
    for col1, col2 in combinations(df_copy.columns, 2):
        if col1 == target or col2 == target:
            continue
        df_copy["interaccion"] = df_copy[col1] * df_copy[col2]
        corr = df_copy["interaccion"].corr(df_copy[target])
        resultados.append((col1, col2, corr))

    return pd.DataFrame(resultados, columns=["Atributo 1", "Atributo 2", "Correlación"]).sort_values(by="Correlación", ascending=False)

def correlacion_triple(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Calcula la correlación entre el target y el producto de cada trío de columnas.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            df_copy[col] = LabelEncoder().fit_transform(df_copy[col].astype(str))

    resultados = []
    for col1, col2, col3 in combinations(df_copy.columns, 3):
        if target in [col1, col2, col3]:
            continue
        df_copy["interaccion"] = df_copy[col1] * df_copy[col2] * df_copy[col3]
        corr = df_copy["interaccion"].corr(df_copy[target])
        resultados.append((col1, col2, col3, corr))

    return pd.DataFrame(resultados, columns=["Atributo 1", "Atributo 2", "Atributo 3", "Correlación"]).sort_values(by="Correlación", ascending=False)

def correlacion_cuadruple(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Calcula la correlación entre el target y el producto de cada combinación de 4 atributos.
    Codifica variables categóricas si es necesario.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            df_copy[col] = LabelEncoder().fit_transform(df_copy[col].astype(str))

    resultados = []
    for col1, col2, col3, col4 in combinations(df_copy.columns, 4):
        if target in [col1, col2, col3, col4]:
            continue
        df_copy["interaccion"] = df_copy[col1] * df_copy[col2] * df_copy[col3] * df_copy[col4]
        corr = df_copy["interaccion"].corr(df_copy[target])
        resultados.append((col1, col2, col3, col4, corr))

    return pd.DataFrame(resultados, columns=["Atributo 1", "Atributo 2", "Atributo 3", "Atributo 4", "Correlación"]).sort_values(by="Correlación", ascending=False)