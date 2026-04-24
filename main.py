#Carga de funciones

from Subida_data import *
from procesamiento_bases import *
from EDA import *
from models import *

data_train=buscar_y_cargar("fraudTrain.csv")
data_test=buscar_y_cargar("fraudTest.csv")
data=pd.concat([data_train, data_test], ignore_index=True)
print(data.head())

# Definir las que NO queremos considerando id y variables que no tenga sentido analizar
cols_a_eliminar = ['Unnamed: 0', 'cc_num', 'unix_time',"is_fraud","zip"]
cols_a_eliminar2 = ["trans_num","trans_date_trans_time"]
# Crear una lista con el nombre de las variables categóricas
cat_columns = data.select_dtypes(include=['object', 'string']).drop(columns=cols_a_eliminar2,errors="ignore").columns

# Creamos la lista de numéricas excluyendo las de arriba
num_columns = data.select_dtypes(include=['number']).drop(columns=cols_a_eliminar, errors='ignore').columns


# ===================================================================
# EJECUTAR ANÁLISIS DEL DATASET
# ===================================================================
#  Identificador único de persona
data['persona_id'] = (
    data['first'].astype(str) + "_" +
    data['last'].astype(str) + "_" +
    data['gender'].astype(str) + "_" +
    data['dob'].astype(str) + "_" +
    data['lat'].astype(str) + "_" +
    data['long'].astype(str)
)

# Cuántos cc_num distintos tiene cada persona
tarjetas_por_persona = (
    data.groupby('persona_id')['cc_num']
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


#-------------------------------------------------------
#  Calcular el mínimo y el máximo de las fechas
#-------------------------------------------------------
#PERIODO DE DATA TEST
fecha_min_test = data_test["trans_date_trans_time"].min()
fecha_max_test = data_test["trans_date_trans_time"].max()

#PERIODO DE DATA_TRAIN
fecha_min_train = data_train["trans_date_trans_time"].min()
fecha_max_train = data_train["trans_date_trans_time"].max()

#PERIODO DE DATA
fecha_min_data = data["trans_date_trans_time"].min()
fecha_max_data = data["trans_date_trans_time"].max()

print("="*40)
print("RANGO TEMPORAL DE DATA")
print("="*40)
print(f"Primera transacción (Mín): {fecha_min_data}")
print(f"Última transacción (Máx):  {fecha_max_data}")

print("="*40)
print("RANGO TEMPORAL DE DATA_TEST")
print("="*40)
print(f"Primera transacción (Mín): {fecha_min_test}")
print(f"Última transacción (Máx):  {fecha_max_test}")

print("="*40)
print("RANGO TEMPORAL DE DATA_TRAIN")
print("="*40)
print(f"Primera transacción (Mín): {fecha_min_train}")
print(f"Última transacción (Máx):  {fecha_max_train}")

#---------------------------------------------------
print("="*40)
print("CONTEO DE COMERCIOS")
print("="*40)
print(data["merchant"].nunique())
print("="*40)
print("CONTEO DE CATEGORY")
print("="*40)
print(data["category"].nunique())
print("="*40)
print("CONTEO DE CITY")
print("="*40)
print(data["city"].nunique())


print("\n" + "="*60)
print("COMENZANDO EL EDA")

print("\n" + "="*60)
print("1. ANÁLISIS DE NULOS")
print("="*60)
ver_nulos(data)

print("\n" + "="*60)
print("2. ANÁLISIS DE DUPLICADOS")
print("="*60)
ver_duplicados(data)


print("\n" + "="*60)
print("3. BALANCE DE CLASES DATA(is_fraud)")
print("="*60)
print(balance_clases(data)) #0.52
print("\n" + "="*60)
print("\n" + "="*60)
print("3. BALANCE DE CLASES DATA_TRAIN(is_fraud)")
print("="*60)
print(balance_clases(data_train)) #0.58%
print("\n" + "="*60)
print("\n" + "="*60)
print("BALANCE DE CLASES DATA_TEST(is_fraud)")
print("="*60)
print(balance_clases(data_test))#0.38%
print("\n" + "="*60)

#----------------------------------------------
#GRÁFICOS
#----------------------------------------------


print("\n" + "="*60)
print("4. RESUMEN ESTADÍSTICO DEL DATASET")
print("="*60)
resumen(data)

print("\n" + "="*60)
print("4. ANÁLISIS UNIVARIADO")
print("="*60)
#make_histogram(data,"amt")
#graficar_densidad(data,num_columns)
#for num_var in num_columns:
  #  make_boxplot(data,num_var)

#for cat_var in cat_columns:
 #   make_barplot(data,cat_var,top=15) #univariado

#------------------------------------------------------
#ANÁLISIS MULTIVARIADO
#------------------------------------------------------
#boxplots_con_tabla(data, num_columns,target="is_fraud")

#make_stacked_barplots(data, cat_columns, top=10)
#print("\n Matriz de correlación...")
#make_heat_map(data_train,num_columns)#multivariado

#graficar_temporalidad_fraude(data_train)
#grafico_tasa_por_variable(data_train, 'category')
#grafico_tasa_por_variable(data_train, 'es_nuevo')# Analizando Riesgo en Comercios Nuevos

#print("Ranking de Categorías más Peligrosas")
#graficar_riesgo_porcategoria(data_train, "category") #en que categorías hay + fraude?

#grafico_tasa_por_variable(data_train, "gender") #influye el genero en la probabilidad?

#print("\n" + "="*70)
#print("EXPLORACIÓN PROFUNDA")
#print("="*70)


#print("\nTabla de estadísticas comparativas")
#tabla = tabla_estadisticas_fraude(data, num_columns)

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
print("7.1 CÁLCULO DE LOGARITMO DE MONTO")
print("="*60)
try:
    data_train["amt_log"] = np.log1p(data_train["amt"])
    print("Logaritmo de monto calculado exitosamente")
    print("Primeros 10 valores de amt_log:")
    print(data_train["amt_log"].head(10))
except Exception as e:
    print(f"Error al calcular amt_log: {e}")

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

print("\n" + "="*60)
print("CÁLCULO DE LOGARITMO DE CITY_POP")
print("="*60)
try:
    data_train["city_pop_log"] = np.log1p(data_train["city_pop"])
    print("Logaritmo de city_pop calculado exitosamente")
    print("Primeros 10 valores:")
    print(data_train["city_pop_log"].head(10))
except Exception as e:
    print(f"Error al calcular city_pop_log: {e}")

print("\n" + "="*60)
print("CÁLCULO DE LOGARITMO DE VELOCIDAD")
print("="*60)
try:
    data_train["velocidad_log"] = np.log1p(data_train["velocidad"])
    print("Logaritmo de velocidad calculado exitosamente")
    print("Primeros 10 valores:")
    print(data_train["velocidad_log"].head(10))
except Exception as e:
    print(f"Error al calcular velocidad_log: {e}")


#TABLAS SIN UNSO ACTUALMENTE PQ NO SE HAN AÑADIDO LAS NUEVAS

# Definir categorías online
#categorias_net = ["grocery_net", "misc_net", "shopping_net"]

# Creo una variable: ES ONLINE (1) VS FÍSICO (0)

#data_train["es_online"] = data_train["category"].isin(categorias_net).astype(int)

#print("Conteo de compras online vs físicas:")
#print(data_train["es_online"].value_counts())

#print("\nPorcentaje:")
#print(data_train["es_online"].value_counts(normalize=True) * 100)

#grafico_tasa_por_variable(data_train, "es_online")

data["distancia"]=obtener_distancia_entre_comercios(data)
print(data["distancia"].tail(10))
columnas_nuevas = ["velocidad_local", "velocidad_internet", "is_first_buy"]
data[columnas_nuevas] = calcular_velocidad(data)

# Verificación opcional
print(data[columnas_nuevas].tail(10))