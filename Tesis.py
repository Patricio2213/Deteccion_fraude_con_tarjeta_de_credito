#Cargar base de datos para entrenamiento


from procesamiento_bases import nuevo_comercio
from Subida_data import buscar_y_cargar
data_train=buscar_y_cargar("fraudTrain.csv")

#Verificar presencia de nulos y duplicados
from procesamiento_bases import ver_nulos
from procesamiento_bases import ver_duplicados
from procesamiento_bases import resumen
from procesamiento_bases import balance_clases
from procesamiento_bases import calcular_edad, calcular_anomaliaencategoria, nuevo_comercio
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
    data_train["velocidad"],data_train["is_first_buy"] = calcular_velocidad(data_train)
    print("Velocidad calculada exitosamente")
    print("Primeras 10 velocidades:")
    print(data_train["velocidad"].head(10))
except Exception as e:
    print(f"Error al calcular velocidad: {e}")
    print("\n" + "=" * 60)


print("\n" + "=" * 60)
print("6. CÁLCULO DE HAVERSINE")
print("=" * 60)
try:
    data_train["distancia"] = haversine(data_train["lat"],data_train["long"],data_train["merch_lat"],data_train["merch_long"])
    print("Distancia calculada exitosamente")
    print("Primeras 10 distancias:")
    print(data_train["distancia"].head(10))
except Exception as e:
    print(f"Error al calcular distancia: {e}")


print("\n" + "="*60)
print("7. CÁLCULO DE Z-SCORE DE MONTO")
print("="*60)
try:
    data_train["zscore"] = zcore_monto(data_train)
    print("Z-score de monto calculado exitosamente")
    print("Primeros 10 z-scores:")
    print(data_train["zscore"].head(10))
except Exception as e:
    print(f"Error al calcular z-score: {e}")

print("\n" + "="*60)
print("8. Cálculo de EDAD del cliente")
print("="*60)
try:
    data_train["edad"] = calcular_edad(data_train)
    print("Un vistazo de 10 edades para confirmar procedimiento")
    print(data_train["edad"].head(10))
except Exception as e:
    print(f"Error al calcular la edad: {e}")

print("\n" + "="*60)
print("9. Cálculo de ANOMALIAS en categoría")
print("="*60)

try:
    data_train["tasa_categoria"] = calcular_anomaliaencategoria(data_train)
    print("10 anomalías en categoría (tasa de habitualidad)")
    print(data_train["tasa_categoria"].head(10))
except Exception as e:
    print(f"Error al calcular la anomalía en categoría: {e}")

print("\n" + "="*60)
print("10. NUEVO COMERCIO")
print("="*60)

try:
    data_train["es_nuevo"]= nuevo_comercio(data_train)
    print("1= nuevo comercio / 0= comercio recurrente o no nuevo")
    print(data_train[["cc_num","merchant", "es_nuevo"]].head(10))
except Exception as e:
    print(f"Error al calcular si es nuevo o no: {e}")

print("\n" + "="*60)
print("11. CONTEO DE REVISIÓN NUEVO COMERCIO")
print("\nCONTEO TOTAL (Frecuencia):")
print(data_train["es_nuevo"].value_counts())
print("\nDISTRIBUCIÓN EN PORCENTAJE (%):")
print(data_train["es_nuevo"].value_counts(normalize=True) * 100)
print("="*60)

#----------------------------------------------
#EDA

from EDA import graficar_densidad, boxplots, graficar_reloj_fraude, graficar_correlacion
from EDA import grafico_tasa_por_variable, graficar_riesgo_porcategoria, separar_columnas
from EDA import tabla_estadisticas_fraude


print("\n" + "="*60)
print("COMENZANDO EL EDA")
num_columns, cat_columns=separar_columnas(data_train)

print("\n" + "="*60)
print("1. RESUMEN ESTADÍSTICO DEL DATASET")
print("="*60)
resumen(data_train)
var_claves_num=["amt","velocidad","distancia","zscore","edad","tasa_categoria"]
var_claves_cat=["gender","merchant","category","city","job","es_nuevo","is_first_buy"]

graficar_densidad(data_train,var_claves_num,target="is_fraud")
boxplots(data_train,var_claves_num,target="is_fraud")
graficar_reloj_fraude(data_train)

grafico_tasa_por_variable(data_train, 'category')
grafico_tasa_por_variable(data_train, 'es_nuevo')# Analizando Riesgo en Comercios Nuevos

print("Ranking de Categorías más Peligrosas")
graficar_riesgo_porcategoria(data_train, "category") #en que categorías hay + fraude?

grafico_tasa_por_variable(data_train, "gender") #influye el genero en la probabilidad?

print("\n" + "="*70)
print("EXPLORACIÓN PROFUNDA")
print("="*70)

print("\n Matriz de correlación...")
graficar_correlacion(data_train, var_claves_num)

print("\nTabla de estadísticas comparativas")
tabla = tabla_estadisticas_fraude(data_train, var_claves_num)

