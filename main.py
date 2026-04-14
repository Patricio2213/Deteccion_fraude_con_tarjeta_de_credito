#Cargar base de datos para entrenamiento
import pandas as pd
import numpy as np

#Carga de funciones

from Subida_data import *
from procesamiento_bases import *
from EDA import *

data_train=buscar_y_cargar("fraudTrain.csv")

# Crear una lista con el nombre de las variables categóricas
cat_columns = data_train.select_dtypes(include=['object', 'string']).columns


# Definir las que NO queremos (basado en tu matriz de correlación)
cols_a_eliminar = ['Unnamed: 0', 'cc_num', 'unix_time',"is_fraud"]

# Creamos la lista de numéricas excluyendo las de arriba
num_columns = data_train.select_dtypes(include=['number']).drop(columns=cols_a_eliminar, errors='ignore').columns


# ===================================================================
# EJECUTAR ANÁLISIS DEL DATASET
# ===================================================================
#  Identificador único de persona
data_train['persona_id'] = (
    data_train['first'].astype(str) + "_" +
    data_train['last'].astype(str) + "_" +
    data_train['gender'].astype(str) + "_" +
    data_train['dob'].astype(str) + "_" +
    data_train['lat'].astype(str) + "_" +
    data_train['long'].astype(str)
)

# Cuántos cc_num distintos tiene cada persona
tarjetas_por_persona = (
    data_train.groupby('persona_id')['cc_num']
    .nunique()
    .reset_index()
)

tarjetas_por_persona.columns = ['persona_id', 'cantidad_tarjetas']

print("\nCantidad de tarjetas por persona:")
print(tarjetas_por_persona.head())

# Filtro de personas con más de una tarjeta
personas_con_multiples_tarjetas = tarjetas_por_persona[
    tarjetas_por_persona['cantidad_tarjetas'] > 1
]

print("\nPersonas con más de una tarjeta:")
print(personas_con_multiples_tarjetas)

# Máximo número de tarjetas que tiene una persona
max_tarjetas = tarjetas_por_persona['cantidad_tarjetas'].max()

print(f"\nMáximo número de tarjetas que tiene una persona: {max_tarjetas}")

print("\n" + "="*60)
print("COMENZANDO EL EDA")

print("\n" + "="*60)
print("1. ANÁLISIS DE NULOS")
print("="*60)
ver_nulos(data_train)

print("\n" + "="*60)
print("2. ANÁLISIS DE DUPLICADOS")
print("="*60)
ver_duplicados(data_train)

print("\n" + "="*60)
print("3. BALANCE DE CLASES (is_fraud)")
print("="*60)
print(balance_clases(data_train)) #0.58%
print("\n" + "="*60)

#----------------------------------------------
#GRÁFICOS
#----------------------------------------------


print("\n" + "="*60)
print("4. RESUMEN ESTADÍSTICO DEL DATASET")
print("="*60)
resumen(data_train)


#graficar_densidad(data_train,num_columns,target="is_fraud")
#for num_var in num_columns:
 #   make_boxplot(data_train,num_var)

#for cat_var in cat_columns:
 #   make_barplot(data_train,cat_var,top=15) #univariado

#print("\n Matriz de correlación...")
#make_heat_map(data_train,num_columns)#multivariado

#graficar_temporalidad_fraude(data_train)

#grafico_tasa_por_variable(data_train, 'category')
#grafico_tasa_por_variable(data_train, 'es_nuevo')# Analizando Riesgo en Comercios Nuevos

#print("Ranking de Categorías más Peligrosas")
#graficar_riesgo_porcategoria(data_train, "category") #en que categorías hay + fraude?

#grafico_tasa_por_variable(data_train, "gender") #influye el genero en la probabilidad?

print("\n" + "="*70)
print("EXPLORACIÓN PROFUNDA")
print("="*70)


print("\nTabla de estadísticas comparativas")
tabla = tabla_estadisticas_fraude(data_train, num_columns)

#make_scatter_plot(data_train,var_claves_num)#multivariado  #DA PROBLEMAS, LO ASOCIO A LA CANTIDAD DE OBSERVACIONES

#make_stacked_barplots(data_train, cat_columns, top=10)#multivariado #NO APORTA MUCHA INFORMACIÓN DADO EL DESBALANCE


#--------------------------------
#ADICIÓN DE NUEVAS COLUMNA
#--------------------------------
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




#TABLAS SIN UNSO ACTUALMENTE PQ NO SE HAN AÑADIDO LAS NUEVAS
#stats_vel_original = data_train.groupby('is_fraud')['velocidad'].describe(percentiles=[.25, .5, .75])

 # Cálculo de los componentes del Boxplot
#stats_vel_original['IQR'] = stats_vel_original['75%'] - stats_vel_original['25%']

# Bigote Superior: Donde estadísticamente empiezan los outliers
#stats_vel_original['Bigote_Superior'] = stats_vel_original['75%'] + (1.5 * stats_vel_original['IQR'])

# Mostrar la tabla
#print("="*65)
#print("TABLA ESTADÍSTICA")
#print("="*65)
#print(stats_vel_original.T)

#tabla_a, tabla_b = generar_tablas_tesis(data_train, var_claves_num)

#print("\n" + "="*80)
#print("TABLA A: ESTADÍSTICAS DESCRIPTIVAS POR GRUPO")
#print("="*80)
#print(tabla_a.to_string(index=False))

#print("\n" + "="*80)
#print("TABLA B: MÉTRICAS DE COMPARACIÓN Y SEPARACIÓN (FRAUDE VS LEGÍTIMO)")
#print("="*80)
#print(tabla_b.to_string(index=False))