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

import numpy as np
import pandas as pd


def haversine_vec(lat1, lon1, lat2, lon2):
    r = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2 +
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return r * c


def distancia_entre_comercios(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    columnas_necesarias = [
        "cc_num", "unix_time", "category",
        "merch_lat", "merch_long", "lat", "long"
    ]

    faltantes = [c for c in columnas_necesarias if c not in df.columns]

    if faltantes:
        raise ValueError(
            f"Faltan columnas para distancia: {faltantes}. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    categorias_net = ["grocery_net", "misc_net", "shopping_net"]

    df["_orden_original"] = np.arange(len(df))

    df["unix_time"] = pd.to_numeric(df["unix_time"], errors="coerce")

    for col in ["lat", "long", "merch_lat", "merch_long"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["cc_num", "unix_time"]).copy()

    df = df.sort_values(["cc_num", "unix_time"]).copy()

    df["es_online"] = df["category"].isin(categorias_net).astype(int)

    df["distancia_local"] = 0.0
    df["distancia_internet"] = 0.0
    df["is_first_local"] = 0
    df["is_first_internet"] = 0

    # =========================
    # COMPRAS LOCALES
    # =========================
    mask_local = df["es_online"] == 0

    prev_lat_local = (
        df.loc[mask_local]
        .groupby("cc_num")["merch_lat"]
        .shift(1)
    )

    prev_lon_local = (
        df.loc[mask_local]
        .groupby("cc_num")["merch_long"]
        .shift(1)
    )

    idx_local = df.loc[mask_local].index

    first_local = prev_lat_local.isna()

    df.loc[idx_local, "is_first_local"] = first_local.astype(int).values

    distancia_local = np.where(
        first_local,
        haversine_vec(
            df.loc[idx_local, "lat"],
            df.loc[idx_local, "long"],
            df.loc[idx_local, "merch_lat"],
            df.loc[idx_local, "merch_long"]
        ),
        haversine_vec(
            prev_lat_local,
            prev_lon_local,
            df.loc[idx_local, "merch_lat"],
            df.loc[idx_local, "merch_long"]
        )
    )

    df.loc[idx_local, "distancia_local"] = distancia_local

    # =========================
    # COMPRAS INTERNET
    # =========================
    mask_online = df["es_online"] == 1

    prev_lat_online = (
        df.loc[mask_online]
        .groupby("cc_num")["merch_lat"]
        .shift(1)
    )

    prev_lon_online = (
        df.loc[mask_online]
        .groupby("cc_num")["merch_long"]
        .shift(1)
    )

    idx_online = df.loc[mask_online].index

    first_online = prev_lat_online.isna()

    df.loc[idx_online, "is_first_internet"] = first_online.astype(int).values

    distancia_online = np.where(
        first_online,
        haversine_vec(
            df.loc[idx_online, "lat"],
            df.loc[idx_online, "long"],
            df.loc[idx_online, "merch_lat"],
            df.loc[idx_online, "merch_long"]
        ),
        haversine_vec(
            prev_lat_online,
            prev_lon_online,
            df.loc[idx_online, "merch_lat"],
            df.loc[idx_online, "merch_long"]
        )
    )

    df.loc[idx_online, "distancia_internet"] = distancia_online

    for col in ["distancia_local", "distancia_internet"]:
        df[col] = (
            df[col]
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
            .clip(lower=0)
        )

    df["is_first_local"] = df["is_first_local"].fillna(0).astype(int)
    df["is_first_internet"] = df["is_first_internet"].fillna(0).astype(int)

    df = df.sort_values("_orden_original").drop(columns=["_orden_original", "es_online"])

    return df


def calcular_velocidad(df):
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    columnas_necesarias = [
        "cc_num",
        "unix_time",
        "distancia_local",
        "distancia_internet",
        "is_first_local",
        "is_first_internet"
    ]

    faltantes = [c for c in columnas_necesarias if c not in df.columns]

    if faltantes:
        raise ValueError(
            f"Faltan columnas para velocidad: {faltantes}. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    df["_orden_original"] = np.arange(len(df))

    df["unix_time"] = pd.to_numeric(df["unix_time"], errors="coerce")

    df = df.dropna(subset=["cc_num", "unix_time"]).copy()

    df = df.sort_values(["cc_num", "unix_time"]).copy()

    df["unix_time_prev"] = df.groupby("cc_num")["unix_time"].shift(1)

    df["delta_tiempo_hours"] = (
        df["unix_time"] - df["unix_time_prev"]
    ) / 3600

    df["delta_tiempo_hours"] = (
        df["delta_tiempo_hours"]
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    df.loc[df["delta_tiempo_hours"] < 0, "delta_tiempo_hours"] = 0

    df["is_first_buy"] = df["unix_time_prev"].isna().astype(int)

    df["delta_tiempo_local"] = 0.0
    df["delta_tiempo_internet"] = 0.0
    df["velocidad_local"] = 0.0
    df["velocidad_internet"] = 0.0

    mask_local = (
        (df["distancia_local"] > 0) &
        (df["is_first_local"] == 0) &
        (df["delta_tiempo_hours"] > 0)
    )

    mask_online = (
        (df["distancia_internet"] > 0) &
        (df["is_first_internet"] == 0) &
        (df["delta_tiempo_hours"] > 0)
    )

    df.loc[mask_local, "delta_tiempo_local"] = df.loc[mask_local, "delta_tiempo_hours"]
    df.loc[mask_online, "delta_tiempo_internet"] = df.loc[mask_online, "delta_tiempo_hours"]

    df.loc[mask_local, "velocidad_local"] = (
        df.loc[mask_local, "distancia_local"] /
        df.loc[mask_local, "delta_tiempo_local"]
    )

    df.loc[mask_online, "velocidad_internet"] = (
        df.loc[mask_online, "distancia_internet"] /
        df.loc[mask_online, "delta_tiempo_internet"]
    )

    tope_velocidad = 1000

    for col in ["velocidad_local", "velocidad_internet"]:
        df[col] = (
            df[col]
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
            .clip(lower=0, upper=tope_velocidad)
        )

    df = df.drop(columns=["unix_time_prev"])

    df = df.sort_values("_orden_original").drop(columns="_orden_original")

    return df

def distancia_cliente_comercio(df):
    """
    Calcula la distancia entre la ubicación del cliente y el comercio
    de cada transacción.

    Crea:
    - d_cliente_comercio_loc: distancia cliente-comercio en compras locales
    - d_cliente_comercio_int: distancia cliente-comercio en compras internet
    """

    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    columnas_necesarias = [
        "category",
        "lat",
        "long",
        "merch_lat",
        "merch_long"
    ]

    faltantes = [c for c in columnas_necesarias if c not in df.columns]

    if faltantes:
        raise ValueError(
            f"Faltan columnas para calcular distancia cliente-comercio: {faltantes}. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    categorias_net = ["grocery_net", "misc_net", "shopping_net"]

    for col in ["lat", "long", "merch_lat", "merch_long"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["es_online"] = df["category"].isin(categorias_net).astype(int)

    distancia = haversine_vec(
        df["lat"],
        df["long"],
        df["merch_lat"],
        df["merch_long"]
    )

    distancia = (
        pd.Series(distancia, index=df.index)
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
        .clip(lower=0)
    )

    df["d_cliente_comercio_loc"] = 0.0
    df["d_cliente_comercio_int"] = 0.0

    df.loc[df["es_online"] == 0, "d_cliente_comercio_loc"] = distancia[df["es_online"] == 0]
    df.loc[df["es_online"] == 1, "d_cliente_comercio_int"] = distancia[df["es_online"] == 1]

    df = df.drop(columns=["es_online"])

    return df

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
