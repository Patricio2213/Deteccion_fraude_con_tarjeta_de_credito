#Paquetes
from pickletools import long1

import numpy as np
import pandas as pd

#Cargar función de carga de archivos
from Subida_data import buscar_y_cargar
data_train=buscar_y_cargar("fraudTrain.csv")
print(data_train.columns)

#Verificar presencia de nulos
data_train.isna().sum()
#Creación de columnas extra obligatorias

#Haversine
def haversine(lat1, lon1, lat2, lon2):
    r=6371 #radio de la tierra en km

    lat1, lon1, lat2, lon1= map(np.radians,[lat1, lon1, lat2, lon2])
    dlat=lat2-lat1
    dlon=lon2-lon1
    a=np.sin(dlat/2)**2+ np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c=2*np.arcsin(np.sqrt(a))
    return r*c

#velocidad de transacciones
def calcular_velocidad(data):
    data=data.sort_values(["cc_num","trans_date_trans_time"])
    #aseguar formato
    data["trans_date_trans_time"]=pd.to_datetime(data["trans_date_trans_time"])
    #tiempo entre transacciones s
    tiempo_horas=(data.groupby("cc_num")["trans_date_trans_time"].diff().dt.total_seconds()/3600)

    #distancia entre transacciones
    distancia_prev= data.groupby("cc_num")["distancia_km"].diff()
    #velocidad
    velocidad=distancia_prev/tiempo_horas
    return velocidad

def zcore_monto(data):
    #prom por tarjeta
    mean_amt=data.groupby("cc_num")["amt"].transform("mean")

    #sd por tarjeta
    std_amt=data.groupby("cc_num")["amt"].transform("std")
    #zcore
    data["zcore_amt"]=(data["amt"]-mean_amt)/std_amt
    return data







