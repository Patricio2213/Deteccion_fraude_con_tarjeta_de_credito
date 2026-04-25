#Paquetes

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


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



#-------------------------------------------------------------------
#DITANCIA

# 1. Definición de la fórmula Haversine (Matemática pura)
def haversine(lat1, lon1, lat2, lon2):
    r = 6371  # Radio de la Tierra en km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c



def haversine(lat1, lon1, lat2, lon2):
    r = 6371  # km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return r * c

def distancia_entre_comercios(df):

    categorias_net = ["grocery_net", "misc_net", "shopping_net"]

    df = df.copy()
    orden_original = df.index

    df = df.sort_values(["cc_num", "unix_time"]).copy()

    df["lat_prev"] = df.groupby("cc_num")["merch_lat"].shift(1)
    df["lon_prev"] = df.groupby("cc_num")["merch_long"].shift(1)
    df["cat_prev"] = df.groupby("cc_num")["category"].shift(1)
    df["unix_time_prev"] = df.groupby("cc_num")["unix_time"].shift(1)

    df["is_first_buy"] = df["unix_time_prev"].isna().astype(int)

    distancia_calc = haversine(
        df["lat_prev"],
        df["lon_prev"],
        df["merch_lat"],
        df["merch_long"]
    ).fillna(0)

    df["distancia_local"] = 0.0
    df["distancia_internet"] = 0.0

    actual_online = df["category"].isin(categorias_net)
    prev_online = df["cat_prev"].isin(categorias_net)

    mask_local = (~actual_online) & (~prev_online) & (df["is_first_buy"] == 0)
    mask_online = actual_online & prev_online & (df["is_first_buy"] == 0)

    df.loc[mask_local, "distancia_local"] = distancia_calc.loc[mask_local]
    df.loc[mask_online, "distancia_internet"] = distancia_calc.loc[mask_online]

    df.drop(
        columns=["lat_prev", "lon_prev", "cat_prev", "unix_time_prev"],
        inplace=True
    )

    return df.loc[orden_original]

def calcular_velocidad(df):

    df = df.copy()
    orden_original = df.index

    df = df.sort_values(["cc_num", "unix_time"]).copy()

    df["unix_time_prev"] = df.groupby("cc_num")["unix_time"].shift(1)

    df["is_first_buy"] = df["unix_time_prev"].isna().astype(int)

    # Tiempo general
    df["delta_tiempo_horas"] = (
        df["unix_time"] - df["unix_time_prev"]
    ) / 3600

    df["delta_tiempo_horas"] = df["delta_tiempo_horas"].fillna(0)

    # 🔥 NUEVO: tiempo separado
    df["delta_tiempo_local"] = 0.0
    df["delta_tiempo_internet"] = 0.0

    mask_tiempo_valido = df["delta_tiempo_horas"] > 0

    mask_local = (df["distancia_local"] > 0) & mask_tiempo_valido
    mask_internet = (df["distancia_internet"] > 0) & mask_tiempo_valido

    df.loc[mask_local, "delta_tiempo_local"] = df.loc[mask_local, "delta_tiempo_horas"]
    df.loc[mask_internet, "delta_tiempo_internet"] = df.loc[mask_internet, "delta_tiempo_horas"]

    # 🔥 Velocidades usando su propio tiempo
    df["velocidad_local"] = 0.0
    df["velocidad_internet"] = 0.0

    mask_local_valido = (df["distancia_local"] > 0) & (df["delta_tiempo_local"] > 0)
    mask_internet_valido = (df["distancia_internet"] > 0) & (df["delta_tiempo_internet"] > 0)

    df.loc[mask_local_valido, "velocidad_local"] = (
        df.loc[mask_local_valido, "distancia_local"] /
        df.loc[mask_local_valido, "delta_tiempo_local"]
    )

    df.loc[mask_internet_valido, "velocidad_internet"] = (
        df.loc[mask_internet_valido, "distancia_internet"] /
        df.loc[mask_internet_valido, "delta_tiempo_internet"]
    )

    # Limpieza
    df["velocidad_local"] = df["velocidad_local"].replace([np.inf, -np.inf], 0).fillna(0)
    df["velocidad_internet"] = df["velocidad_internet"].replace([np.inf, -np.inf], 0).fillna(0)

    df.drop(columns=["unix_time_prev"], inplace=True)

    return df.loc[orden_original]

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

#Procesador para los modelos
def get_preprocessor():
    # 1. Definir columnas que usaremos de cada tipo
    numeric_features = ['amt', 'city_pop']
    categorical_features = ['category', 'state']

    # 2. Transformador
    preprocessor = ColumnTransformer(
        transformers=[
            # Estandarizar variables numéricas
            ('num', StandardScaler(), numeric_features),
            # Convertir categóricas a dummies (ignora categorías nuevas en producción)
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    return preprocessor
