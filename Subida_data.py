import pandas as pd
from pathlib import Path

def buscar_y_cargar(nombre_archivo):
    ruta_base = Path(__file__).resolve().parent
    ruta_final = ruta_base / "archive" / nombre_archivo
    print("Cargando desde:", ruta_final)
    return pd.read_csv(ruta_final)

##Probando formula
data_train = buscar_y_cargar("fraudTrain.csv")
print(data_train.head())  # Descomenta esta línea para ver las primeras filas

#Cargar también el archivo de test:
data_test = buscar_y_cargar("fraudTest.csv")
print(data_test.head())

