#Cargar base de datos para entrenamiento
from Subida_data import buscar_y_cargar
data_train=buscar_y_cargar("fraudTrain.csv")

#Adición de columnas distancia_km, velocidad_kmh
from procesamiento_bases import haversine
from procesamiento_bases import calcular_velocidad
data_train["distancia_km"]= haversine(data_train["lat"],data_train["long"],data_train["merch_lat"],data_train["merch_long"])
#COMPROBACION
#print(data_train["distancia_km"].head())

data_train["velocidad_kmh"]=calcular_velocidad(data_train)
#COMPROBACION
print(data_train["velocidad_kmh"].head())
#GENERA NaN, REPARAR EL CODIGO PARA CUBRIR CUANDO HAY SOLO 1 TRANSACCION

