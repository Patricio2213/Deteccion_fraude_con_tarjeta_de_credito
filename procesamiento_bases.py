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

def haversine(lat1, lon1, lat2, lon2):
    r = 6371  # km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return r * c

def distancia_entre_comercios(df):

    # categorías consideradas como compras online
    categorias_net = ["grocery_net", "misc_net", "shopping_net"]

    # copia el dataframe para no modificar el original
    df = df.copy()

    # guarda el orden original de las filas
    orden_original = df.index

    # ordena las transacciones por tarjeta y tiempo en orden ascendente
    df = df.sort_values(["cc_num", "unix_time"]).copy()

    # crea columnas de salida
    df["distancia_local"] = 0.0
    df["distancia_internet"] = 0.0

    # flags para identificar cuándo se usó el punto ancla cliente → comercio
    df["is_first_local"] = 0
    df["is_first_internet"] = 0

    # función que procesa cada tarjeta por separado
    def procesar_tarjeta(grupo):

        # última compra local registrada para esa tarjeta
        last_local = None

        # última compra online registrada para esa tarjeta
        last_online = None

        # listas para guardar resultados
        resultados_local = []
        resultados_online = []
        flags_first_local = []
        flags_first_internet = []

        # recorre cada transacción de la tarjeta
        for _, row in grupo.iterrows():

            # determina si la compra actual es online
            actual_online = row["category"] in categorias_net

            # coordenadas del comercio actual
            lat_c = row["merch_lat"]
            lon_c = row["merch_long"]

            # coordenadas del cliente / punto ancla
            lat_cliente = row["lat"]
            lon_cliente = row["long"]

            # caso compra online
            if actual_online:

                # si existe una compra online previa
                if last_online is not None:
                    lat_prev, lon_prev = last_online
                    dist = haversine(lat_prev, lon_prev, lat_c, lon_c)
                    flag_first_online = 0

                # si no existe compra online previa, usa punto ancla
                else:
                    dist = haversine(lat_cliente, lon_cliente, lat_c, lon_c)
                    flag_first_online = 1

                resultados_local.append(0.0)
                resultados_online.append(dist)

                flags_first_local.append(0)
                flags_first_internet.append(flag_first_online)

                # actualiza última compra online
                last_online = (lat_c, lon_c)

            # caso compra local
            else:

                # si existe una compra local previa
                if last_local is not None:
                    lat_prev, lon_prev = last_local
                    dist = haversine(lat_prev, lon_prev, lat_c, lon_c)
                    flag_first_local = 0

                # si no existe compra local previa, usa punto ancla
                else:
                    dist = haversine(lat_cliente, lon_cliente, lat_c, lon_c)
                    flag_first_local = 1

                resultados_local.append(dist)
                resultados_online.append(0.0)

                flags_first_local.append(flag_first_local)
                flags_first_internet.append(0)

                # actualiza última compra local
                last_local = (lat_c, lon_c)

        # asigna resultados al grupo
        grupo["distancia_local"] = resultados_local
        grupo["distancia_internet"] = resultados_online
        grupo["is_first_local"] = flags_first_local
        grupo["is_first_internet"] = flags_first_internet

        return grupo

    # aplica la función a cada tarjeta
    df = df.groupby("cc_num", group_keys=False).apply(procesar_tarjeta)

    # evita posibles infinitos o nulos
    df["distancia_local"] = df["distancia_local"].replace([np.inf, -np.inf], 0).fillna(0)
    df["distancia_internet"] = df["distancia_internet"].replace([np.inf, -np.inf], 0).fillna(0)

    # devuelve el dataframe en el orden original
    return df.loc[orden_original]


def calcular_velocidad(df):

    df = df.copy()

    orden_original = df.index

    df = df.sort_values(["cc_num", "unix_time"]).copy()

    df["unix_time_prev"] = df.groupby("cc_num")["unix_time"].shift(1)

    df["is_first_buy"] = df["unix_time_prev"].isna().astype(int)

    df["delta_tiempo_horas"] = (
        df["unix_time"] - df["unix_time_prev"]
    ) / 3600

    df["delta_tiempo_horas"] = df["delta_tiempo_horas"].fillna(0)

    df.loc[df["delta_tiempo_horas"] < 0, "delta_tiempo_horas"] = 0


    df["delta_tiempo_local"] = 0.0
    df["delta_tiempo_internet"] = 0.0

    mask_tiempo_valido = df["delta_tiempo_horas"] > 0

    mask_local = (
        (df["distancia_local"] > 0) &
        (df["is_first_local"] == 0) &
        mask_tiempo_valido
    )

    mask_internet = (
        (df["distancia_internet"] > 0) &
        (df["is_first_internet"] == 0) &
        mask_tiempo_valido
    )

    df.loc[mask_local, "delta_tiempo_local"] = df.loc[mask_local, "delta_tiempo_horas"]

    df.loc[mask_internet, "delta_tiempo_internet"] = df.loc[mask_internet, "delta_tiempo_horas"]

    df["velocidad_local"] = 0.0
    df["velocidad_internet"] = 0.0

    df.loc[mask_local, "velocidad_local"] = (
        df.loc[mask_local, "distancia_local"] /
        df.loc[mask_local, "delta_tiempo_local"]
    )

    df.loc[mask_internet, "velocidad_internet"] = (
        df.loc[mask_internet, "distancia_internet"] /
        df.loc[mask_internet, "delta_tiempo_internet"]
    )

    df["velocidad_local"] = (
        df["velocidad_local"]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    df["velocidad_internet"] = (
        df["velocidad_internet"]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

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

def calcular_anomaliaencategoria(data):

    df = data.copy()

    orden_original = df.index

    df = df.sort_values(["cc_num", "unix_time"]).copy()

    df["total_prev"] = df.groupby("cc_num").cumcount()

    df["cat_prev"] = df.groupby(["cc_num", "category"]).cumcount()

    df["tasa_cliente"] = df["cat_prev"] / df["total_prev"]

    df["tasa_cliente"] = (
        df["tasa_cliente"]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    proporciones_globales = df["category"].value_counts(normalize=True)

    df["tasa_global"] = df["category"].map(proporciones_globales)

    df["tasa_global"] = df["tasa_global"].replace(0, 1e-6)

    df["anomalia_categoria"] = df["tasa_cliente"] / df["tasa_global"]

    df["anomalia_categoria"] = np.log1p(df["anomalia_categoria"])

    df["anomalia_categoria"] = (
        df["anomalia_categoria"]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    return df.loc[orden_original, "anomalia_categoria"]



def nuevo_comercio(data, meses_calentamiento=3):

    data = data.sort_values(["cc_num", "trans_date_trans_time"])

    data["trans_date_trans_time"] = pd.to_datetime(data["trans_date_trans_time"])

    conteo_acum = data.groupby(["cc_num", "merchant"]).cumcount()

    es_nuevo = (conteo_acum == 0).astype(int)

    fecha_inicio = data["trans_date_trans_time"].min()

    fecha_limite = fecha_inicio + pd.Timedelta(days=meses_calentamiento * 30)

    data["is_warmup"] = (data["trans_date_trans_time"] < fecha_limite).astype(int)

    return es_nuevo

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
