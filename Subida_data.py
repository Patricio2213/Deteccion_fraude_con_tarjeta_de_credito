import pandas as pd
import os

def buscar_y_cargar(nombre_archivo):
    ruta_final = os.path.join("archive", nombre_archivo)
    return pd.read_csv(ruta_final)

##Probando formula
data_train=buscar_y_cargar("fraudTrain.csv")
#print(data_test.head())



