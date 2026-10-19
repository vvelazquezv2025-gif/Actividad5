# Funciones para preparar los datos del proyecto Actividad5

import pandas as pd
from sklearn.model_selection import train_test_split


def cargar_datos(ruta_csv):
    """Carga el dataset preparado desde un archivo CSV."""
    return pd.read_csv(ruta_csv)


def separar_datos(df, columna_objetivo="diagnostico"):
    """
    Separa las variables predictoras y la variable objetivo.
    Reserva el 20% de los datos para la prueba final.
    """
    X = df.drop(columns=[columna_objetivo])
    y = df[columna_objetivo]

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
