#Cargar base de datos para entrenamiento
from Subida_data import buscar_y_cargar
data_train=buscar_y_cargar("fraudTrain.csv")

#Verificar presencia de nulos y duplicados
from procesamiento_bases import ver_nulos
from procesamiento_bases import ver_duplicados
print("NULOS POR COLUMNA")
ver_nulos(data_train)

print("\n CANTIDAD DUPLICADOS")
ver_duplicados(data_train)

print("\n RESUMEN DE DATOS")
from procesamiento_bases import resumen
resumen(data_train)

print("\n CANTIDAD DE FRAUDES")
from procesamiento_bases import balance_clases
print(balance_clases(data_train)) #0.58% de fraude


#Adición de columnas distancia_km, velocidad_kmh
from procesamiento_bases import haversine, ver_duplicados
from procesamiento_bases import calcular_velocidad
data_train["distancia_km"]= haversine(data_train["lat"],data_train["long"],data_train["merch_lat"],data_train["merch_long"])
#COMPROBACION
#print(data_train["distancia_km"].head())

data_train["velocidad_kmh"]=calcular_velocidad(data_train)
#COMPROBACION
#print(data_train["velocidad_kmh"])
#ZCORE MONTO
from procesamiento_bases import zcore_monto
data_train["monto_zcore"]=zcore_monto(data_train)

##EDA
#separar columnas
from EDA import separar_columnas
num_columns, cat_columns=separar_columnas(data_train)
#Histogramas
from EDA import histogramas
#var_claves_num=["amt","velocidad_kmh","distancia_km","monto_zcore"]
#histogramas(data_train,var_claves_num)
print(data_train["velocidad_kmh"])


