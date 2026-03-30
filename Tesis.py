#Cargar base de datos para entrenamiento
from Subida_data import buscar_y_cargar
data_train=buscar_y_cargar("fraudTrain.csv")

#Verificar presencia de nulos y duplicados
from procesamiento_bases import ver_nulos
from procesamiento_bases import ver_duplicados
from procesamiento_bases import resumen
from procesamiento_bases import balance_clases

# ===================================================================
# EJECUTAR ANÁLISIS DEL DATASET
# ===================================================================

print("\n" + "="*60)
print("1. ANÁLISIS DE NULOS")
print("="*60)
ver_nulos(data_train)

print("\n" + "="*60)
print("2. ANÁLISIS DE DUPLICADOS")
print("="*60)
ver_duplicados(data_train)

print("\n" + "="*60)
print("3. RESUMEN ESTADÍSTICO DEL DATASET")
print("="*60)
resumen(data_train)

print("\n" + "="*60)
print("4. BALANCE DE CLASES (is_fraud)")
print("="*60)
print(balance_clases(data_train)) #0.58%


#Adición de columnas distancia_km, velocidad_kmh

from procesamiento_bases import haversine
from procesamiento_bases import calcular_velocidad
from procesamiento_bases import zcore_monto
print("\n" + "="*60)
print("5. CÁLCULO DE VELOCIDAD DE TRANSACCIONES")
print("="*60)
try:
    data_train["velocidad"] = calcular_velocidad(data_train)
    print("Velocidad calculada exitosamente")
    print("Primeras 10 velocidades:")
    print(data_train["velocidad"].head(10))
except Exception as e:
    print(f"Error al calcular velocidad: {e}")
    print("\n" + "=" * 60)
    print("5. CÁLCULO DE VELOCIDAD DE TRANSACCIONES")
    print("=" * 60)

print("\n" + "=" * 60)
print("5. CÁLCULO DE HAVERSINE")
print("=" * 60)
try:
    data_train["haversine"] = haversine(data_train["lat"],data_train["long"],data_train["merch_lat"],data_train["merch_long"])
    print("Haversine calculada exitosamente")
    print("Primeras 10 distancias:")
    print(data_train["haversine"].head(10))
except Exception as e:
        print(f"Error al calcular haversine: {e}")


print("\n" + "="*60)
print("6. CÁLCULO DE Z-SCORE DE MONTO")
print("="*60)
try:
    data_train["zscore"] = zcore_monto(data_train)
    print("Z-score de monto calculado exitosamente")
    print("Primeros 10 z-scores:")
    print(data_train["zscore"].head(10))
except Exception as e:
    print(f"Error al calcular z-score: {e}")





##EDA
#separar columnas
from EDA import separar_columnas
num_columns, cat_columns=separar_columnas(data_train)
#Histogramas
from EDA import histogramas
#var_claves_num=["amt","velocidad_kmh","distancia_km","monto_zcore"]
#histogramas(data_train,var_claves_num)
#print(data_train["velocidad"])
