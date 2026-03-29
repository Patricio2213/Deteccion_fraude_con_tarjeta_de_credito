#Paquetes


import numpy as np
import pandas as pd

#Cargar función de carga de archivos
from Subida_data import buscar_y_cargar
#data_train=buscar_y_cargar("fraudTrain.csv")
#print(data_train.columns)

#Verificar presencia de nulos
def ver_nulos(df):
    nulos=df.isna().sum()
    print(nulos)

#Verificar presencia de duplicados
def ver_duplicados(df):
    duplic=df.duplicated().sum()
    print(duplic)

def resumen(df):
    resum=df.describe()
    print(resum)

def balance_clases(df, columna_objetivo='is_fraud'):

        return df[columna_objetivo].value_counts()

#Creación de columnas extra obligatorias

#Haversine
def haversine(lat1, lon1, lat2, lon2):
    r=6371 #radio de la tierra en km

    lat1, lon1, lat2, lon2= map(np.radians,[lat1, lon1, lat2, lon2])
    dlat=lat2-lat1
    dlon=lon2-lon1
    a=np.sin(dlat/2)**2+ np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c=2*np.arcsin(np.sqrt(a))
    return r*c

#velocidad de transacciones
def calcular_velocidad(data):

    # 1. Ordenar por tarjeta y tiempo
    data = data.sort_values(["cc_num", "trans_date_trans_time"])

    # 2. Asegurar formato datetime
    data["trans_date_trans_time"] = pd.to_datetime(data["trans_date_trans_time"])

    # 3. Tiempo entre transacciones (en horas)
    tiempo_horas = (
        data.groupby("cc_num")["trans_date_trans_time"]
        .diff()
        .dt.total_seconds() / 3600
    )

    # 4. Coordenadas anteriores
    data["lat_prev"] = data.groupby("cc_num")["lat"].shift()
    data["lon_prev"] = data.groupby("cc_num")["long"].shift()

    # 5. Distancia real (Haversine)
    data["distancia_km"] = haversine(
        data["lat_prev"],
        data["lon_prev"],
        data["lat"],
        data["long"]
    )

    # 6. Velocidad
    velocidad = data["distancia_km"] / tiempo_horas

    #  Limpieza, deja en 0 valores infinitos, es decir velocidades donde el tiempo entre transacciones era 0
    #velocidad = velocidad.replace([np.inf, -np.inf], np.nan)
    #usarlo podría ocultar el comportamiento anomalo
    # Cambiar por NaN valores extremos----cambiar esto en 0 esconde caracteristicas de fraudes
    #velocidad[velocidad > 1000] = np.nan
    velocidad[velocidad < 0] = np.nan  # por seguridad, se mantiene pq no pueden ser menores de 0

    # 8. Reemplazar NaN por 0
    velocidad = velocidad.fillna(0)

    return velocidad


def zcore_monto(data):
    # 1. prom por tarjeta
    mean_amt = data.groupby("cc_num")["amt"].transform("mean")

    # 2. sd por tarjeta
    std_amt = data.groupby("cc_num")["amt"].transform("std")

    # 3. zcore_monto
    resultado = (data["amt"] - mean_amt) / std_amt

    return resultado








