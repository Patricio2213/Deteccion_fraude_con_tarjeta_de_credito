#Paquetes

import numpy as np
import pandas as pd


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
    haversine=r*c
    return haversine

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

    # 4. Coordenadas anteriores de comercios(distancia entre casa es siempre igual por tarjeta)

    data["merch_lat_prev"] = data.groupby("cc_num")["merch_lat"].shift()
    data["merch_lon_prev"] = data.groupby("cc_num")["merch_long"].shift()

    # 5. Distancia real (Haversine)
    data["distancia_km"] = haversine(
        data["merch_lat_prev"],
        data["merch_lon_prev"],
        data["merch_lat"],
        data["merch_long"]
    )

    # 6. Velocidad
    velocidad = data["distancia_km"] / tiempo_horas

    #7.Crear columna para primeras compras
    is_first_buy = velocidad.isna().astype(int)
    #  Limpieza, dejar infinitos como un numero gigante para notar la anomalia
    velocidad = velocidad.replace([np.inf, -np.inf], 10000)

    velocidad[velocidad < 0] = np.nan  # por seguridad, se mantiene pq no pueden ser menores de 0

    # 8. Reemplazar NaN por -1, esto indica que para primeras transacciones la velocidad será -1
    velocidad = velocidad.fillna(-1)

    return velocidad, is_first_buy


def zcore_monto(data):
    # 1. prom por tarjeta
    mean_amt = data.groupby("cc_num")["amt"].transform("mean")

    # 2. sd por tarjeta
    std_amt = data.groupby("cc_num")["amt"].transform("std")

    # 3. zcore_monto
    resultado = (data["amt"] - mean_amt) / std_amt

    return resultado

#POSIBLES NUEVAS VARIABLES PARA NUESTRO ESTUDIO
def calcular_edad(data): #analizar si ciertos rangos de edad son más vulnerables al fraude
    trans_time = pd.to_datetime(data["trans_date_trans_time"])
    dob = pd.to_datetime(data["dob"])
    edad = (trans_time - dob).dt.days // 365 #así calculo la edad
    return edad #podemos crear un perfil de riesgo demográfico que el modelo podría aprender a segmentar

def calcular_anomaliaencategoria(data): #en vez de ver cuanto gastó, podemos ver en qué gastó
    total_compras = data.groupby("cc_num")["cc_num"].transform("count")
    compras_x_categoria = data.groupby(["cc_num","category"])["category"].transform("count")
    tasa_categoria = compras_x_categoria / total_compras
    return tasa_categoria #buscar ruptura en los hábitos de consumo
#Comentario(pato): Mas que en que gastó no sería en que tipo de local se gastó (Importante de diferenciar en el analisis)

def nuevo_comercio(data, meses_calentamiento=3): #estudiar si el fraude suele ocurrir en establecimientos donde el cliente no ha comprado antes
    data = data.sort_values(["cc_num", "trans_date_trans_time"])
    data["trans_date_trans_time"] = pd.to_datetime(data["trans_date_trans_time"])
    conteo_acum = data.groupby(["cc_num","merchant"]).cumcount()
    es_nuevo= (conteo_acum ==0).astype(int)
    fecha_inicio = data["trans_date_trans_time"].min()
    fecha_limite = fecha_inicio + pd.Timedelta(days=meses_calentamiento *30) #estoy evitando el cold start
    es_nuevo.loc[data["trans_date_trans_time"] < fecha_limite] = -1 #periodo calentamiento 3 meses (-1) si es 1ra vez
    return es_nuevo #será binario 1/0 indicando si es primera vez o no.

