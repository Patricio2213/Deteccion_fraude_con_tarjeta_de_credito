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
def obtener_distancia_entre_comercios(df):

    # 2. Ordenar datos para que el cálculo sea cronológico por usuario
    # Creamos una copia interna para no desordenar tu df original
    df_temp = df.sort_values(by=['cc_num', 'unix_time']).copy()

    # 3. Traer las coordenadas del comercio anterior
    df_temp['lat_com_ant'] = df_temp.groupby('cc_num')['merch_lat'].shift(1)
    df_temp['lon_com_ant'] = df_temp.groupby('cc_num')['merch_long'].shift(1)

    # 4. Definir filtro de categorías (Excluir 1, 2, 3)
    # Solo es válida si la actual NO es online Y la anterior tampoco
    cats_excluir = ["grocery_net", "misc_net", "shopping_net"]
    valida_actual = ~df_temp['category'].isin(cats_excluir)
    valida_anterior = ~df_temp.groupby('cc_num')['category'].shift(1).isin(cats_excluir)

    filtro_fisico = valida_actual & valida_anterior

    # 5. Inicializar la columna de resultados en 0.0
    distancia_serie = pd.Series(0.0, index=df_temp.index)

    # 6. Aplicar la fórmula SOLO a las filas que cumplen el filtro
    df_filtrado = df_temp[filtro_fisico]

    if not df_filtrado.empty:
        distancia_serie.loc[df_filtrado.index] = haversine(
            df_filtrado['lat_com_ant'],
            df_filtrado['lon_com_ant'],
            df_filtrado['merch_lat'],
            df_filtrado['merch_long']
        )

    # 7. Retornar la columna alineada con el índice original
    # (Usamos fillna por si acaso la primera transacción genera un nulo)
    return distancia_serie
#velocidad de transacciones
def calcular_velocidad(data):
    # 0. Función Haversine interna para que no dependa de nada externo
    def haversine(lat1, lon1, lat2, lon2):
        r = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return r * c

    categorias_net = ["grocery_net", "misc_net", "shopping_net"]

    # Trabajamos sobre una copia y ordenamos cronológicamente
    df = data.sort_values(["cc_num", "trans_date_trans_time"]).copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

    # Identificar tipo de transacción
    df["es_online"] = df["category"].isin(categorias_net)

    # --- VELOCIDAD LOCAL ---
    df_local = df[~df["es_online"]].copy()

    if not df_local.empty:
        df_local["tiempo_horas"] = (
            df_local.groupby("cc_num")["trans_date_trans_time"]
            .diff()
            .dt.total_seconds() / 3600
        )

        df_local["lat_prev"] = df_local.groupby("cc_num")["merch_lat"].shift()
        df_local["lon_prev"] = df_local.groupby("cc_num")["merch_long"].shift()

        df_local["distancia_km"] = haversine(
            df_local["lat_prev"], df_local["lon_prev"],
            df_local["merch_lat"], df_local["merch_long"]
        )

        df_local["velocidad_local"] = df_local["distancia_km"] / df_local["tiempo_horas"]
        df_local["velocidad_local"] = df_local["velocidad_local"].replace([np.inf, -np.inf], 0)

    # --- VELOCIDAD INTERNET ---
    df_online = df[df["es_online"]].copy()

    if not df_online.empty:
        df_online["tiempo_horas"] = (
            df_online.groupby("cc_num")["trans_date_trans_time"]
            .diff()
            .dt.total_seconds() / 3600
        )

        df_online["lat_prev"] = df_online.groupby("cc_num")["merch_lat"].shift()
        df_online["lon_prev"] = df_online.groupby("cc_num")["merch_long"].shift()

        df_online["distancia_km"] = haversine(
            df_online["lat_prev"], df_online["lon_prev"],
            df_online["merch_lat"], df_online["merch_long"]
        )

        df_online["velocidad_internet"] = df_online["distancia_km"] / df_online["tiempo_horas"]
        df_online["velocidad_internet"] = df_online["velocidad_internet"].replace([np.inf, -np.inf], 0)

    # --- ASIGNACIÓN DE RESULTADOS AL DATAFRAME PRINCIPAL ---
    df["velocidad_local"] = df_local["velocidad_local"] if not df_local.empty else 0
    df["velocidad_internet"] = df_online["velocidad_internet"] if not df_online.empty else 0

    # Primera compra
    df["is_first_buy"] = (
        df.groupby("cc_num")["trans_date_trans_time"]
        .diff()
        .isna()
        .astype(int)
    )

    df["velocidad_local"] = df["velocidad_local"].fillna(0)
    df["velocidad_internet"] = df["velocidad_internet"].fillna(0)

    return df.sort_index()[["velocidad_local", "velocidad_internet", "is_first_buy"]]
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