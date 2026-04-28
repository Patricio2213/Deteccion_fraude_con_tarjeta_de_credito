#Carga de funciones
from Subida_data import *
from procesamiento_bases import *
from EDA import *
from models import *

#Carga de bases
data_train=buscar_y_cargar("fraudTrain.csv")
data_test=buscar_y_cargar("fraudTest.csv")
data=pd.concat([data_train, data_test], ignore_index=True)
#print(data.head())

# Definir las que NO queremos considerando id y variables que no tengan sentido utilizar en base al EDA
#cols_a_eliminar = ['Unnamed: 0', 'cc_num', 'unix_time',"is_fraud","zip"]
#cols_a_eliminar2 = ["trans_num","trans_date_trans_time"]

# Crear una lista con el nombre de las variables categóricas
#cat_columns = data.select_dtypes(include=['object', 'string']).drop(columns=cols_a_eliminar2,errors="ignore").columns

# Creamos la lista de numéricas excluyendo las de arriba
#num_columns = data.select_dtypes(include=['number']).drop(columns=cols_a_eliminar, errors='ignore').columns

# EJECUTAR ANÁLISIS DEL DATASET

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

print("\nMáximo número de tarjetas por persona:")
#print(tarjetas_por_persona["cantidad_tarjetas"].max())

print("\nRegistros unicos por variable:")
print(data.nunique())

#  Calcular el mínimo y el máximo de las fechas

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
print(f"Primera transacción (Mín): {fecha_min_data}")
print(f"Última transacción (Máx):  {fecha_max_data}")

print("="*40)
print("RANGO TEMPORAL DE DATA_TEST")
print(f"Primera transacción (Mín): {fecha_min_test}")
print(f"Última transacción (Máx):  {fecha_max_test}")

print("="*40)
print("RANGO TEMPORAL DE DATA_TRAIN")
print(f"Primera transacción (Mín): {fecha_min_train}")
print(f"Última transacción (Máx):  {fecha_max_train}")

#---------------------------------------------------
print("="*40)
print("CONTEO DE COMERCIOS")
print(data["merchant"].nunique())
print("="*40)
print("CONTEO DE CATEGORY")
print(data["category"].nunique())
print("="*40)
print("CONTEO DE CITY")
print(data["city"].nunique())
print("\n" + "="*60)
print("COMENZANDO EL EDA")
print("\n" + "="*60)
print("Análisis de nulos")
#ver_nulos(data)
print("\n" + "="*60)
print("Análisis de duplicados")
ver_duplicados(data)
print("\n" + "="*60)
print("Balance de clases base completa (is_fraud)")
print(balance_clases(data)) #0.52
print("\n" + "="*60)
print("Balance de clases DATA_TRAIN(is_fraud)")
print(balance_clases(data_train)) #0.58%
print("\n" + "="*60)
print("Balance de clases DATA_TEST(is_fraud)")
print(balance_clases(data_test))#0.38%

#Gráficos

print("\n" + "="*60)
print("Resumen estadístico")
#resumen(data)

print("\n" + "="*60)
print("Análisis Univariado")
#make_histogram(data,"amt")
#graficar_densidad(data,num_columns)
#for num_var in num_columns:
  #  make_boxplot(data,num_var)

#for cat_var in cat_columns:
#make_barplot(data,"category",top=20) #univariado

#ANÁLISIS MULTIVARIADO

#boxplots_con_tabla(data, num_columns,target="is_fraud")

#make_stacked_barplots(data, cat_columns, top=10)
#print("\n Matriz de correlación...")
#make_heat_map(data,num_columns)#multivariado

#graficar_temporalidad_fraude(data)
#grafico_tasa_por_variable(data, 'category')
#grafico_tasa_por_variable(data, 'es_nuevo')# Analizando Riesgo en Comercios Nuevos

#print("Ranking de Categorías más Peligrosas")
#graficar_riesgo_porcategoria(data, "category") #en que categorías hay + fraude?

#grafico_tasa_por_variable(data, "gender") #influye el genero en la probabilidad?

#print("\n" + "="*70)
#print("EXPLORACIÓN PROFUNDA")

#print("\nTabla de estadísticas comparativas")
#tabla = tabla_estadisticas_fraude(data, num_columns)

#make_scatter_plot(data,var_claves_num)#multivariado  #DA PROBLEMAS, LO ASOCIO A LA CANTIDAD DE OBSERVACIONES

#make_stacked_barplots(data, cat_columns, top=10)#multivariado #NO APORTA MUCHA INFORMACIÓN DADO EL DESBALANCE


# Creación de nuevas variables

data = distancia_entre_comercios(data)
data = calcular_velocidad(data)
data = distancia_cliente_comercio(data)

print("\n" + "="*60)
print("\nÚltimos 3 casos INTERNET:")
print(
    data[
        data["distancia_internet"] > 0
    ][
        [
            "cc_num",
            "distancia_internet",
            "delta_tiempo_internet",
            "velocidad_internet"
        ]
    ].tail(3).to_string()
)

print("\n" + "="*60)
print("\nÚltimos 3 casos LOCAL:")
print(
    data[
        data["distancia_local"] > 0
    ][
        [
            "cc_num",
            "distancia_local",
            "delta_tiempo_local",
            "velocidad_local"
        ]
    ].tail(3).to_string()
)

print("\n" + "="*60)
print("Cálculo de z-score de amt")
try:
    data["zscore"] = zcore_monto(data)
    print("Z-score de monto calculado exitosamente")
    print("Primeros 3 z-scores:")
    print(data["zscore"].head(3))
except Exception as e:
    print(f"Error al calcular z-score: {e}")

print("\n" + "="*60)
print("Edad del cliente")
try:
    data["edad"] = calcular_edad(data)
    print("Un vistazo de 3 edades para confirmar procedimiento")
    print(data["edad"].head(3))
except Exception as e:
    print(f"Error al calcular la edad: {e}")

print("\n" + "="*60)
print("Cálculo de anomalías por categoría (tasa de habitualidad)")
try:
    data["tasa_categoria"] = calcular_anomaliaencategoria(data)
    print("Ultimas 3 anomalías en categoría")
    print(data["tasa_categoria"].tail(3))
except Exception as e:
    print(f"Error al calcular la anomalía en categoría: {e}")

print("\n" + "="*60)
print("Nuevo comercio")
try:
    data["es_nuevo"]= nuevo_comercio(data)
    print(data[["cc_num","merchant", "es_nuevo"]].head(2))
except Exception as e:
    print(f"Error al calcular si es nuevo o no: {e}")

print("\n" + "="*60)
print("Distancia cliente-comercio")
print(
    data[[
        "cc_num",
        "d_cliente_comercio_loc",
        "d_cliente_comercio_int"
    ]]
    .head(3)
    .to_string()
)
#TABLAS SIN UNSO ACTUALMENTE PQ NO SE HAN AÑADIDO LAS NUEVAS

# Definir categorías online
#categorias_net = ["grocery_net", "misc_net", "shopping_net"]

# Creo una variable: ES ONLINE (1) VS FÍSICO (0)

#data["es_online"] = data["category"].isin(categorias_net).astype(int)

#print("Conteo de compras online vs físicas:")
#print(data["es_online"].value_counts())

#print("\nPorcentaje:")
#print(data["es_online"].value_counts(normalize=True) * 100)

#grafico_tasa_por_variable(data, "es_online")

print("\n" + "="*60)
print("Investiguemos que variables podrían necesitar logaritmo")

# Seleccionar solo variables numéricas
num_cols = data.select_dtypes(include="number").columns

resultados = []

for col in num_cols:
    serie = data[col].dropna()

    # Skew original
    skew_original = serie.skew()

    # Skew con log
    if (serie >= 0).all():
        skew_log = np.log1p(serie).skew()
    else:
        skew_log = np.nan  # no se puede aplicar log

    resultados.append([col, skew_original, skew_log])

# Crear DataFrame
tabla_skew = pd.DataFrame(
    resultados,
    columns=["variable", "skew_original", "skew_log"]
)

# Ordenar por mayor skew
tabla_skew = tabla_skew.sort_values(by="skew_original", ascending=False)

# Mostrar tabla completa
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

print(tabla_skew.to_string(index=False))

print("\n" + "="*60)
print("Análisis de Curtosis: Original vs Logaritmo (log1p)")

# Seleccionar solo variables numéricas
num_cols = data.select_dtypes(include="number").columns

resultados_kurtosis = []

for col in num_cols:
    serie = data[col].dropna()

    # Calcular Kurtosis original
    kurt_original = serie.kurtosis()

    # Calcular Kurtosis con logaritmo
    if (serie >= 0).all():
        kurt_log = np.log1p(serie).kurtosis()
    else:
        kurt_log = np.nan

    resultados_kurtosis.append([col, kurt_original, kurt_log])

# Crear DataFrame para la tabla de resultados
tabla_kurtosis = pd.DataFrame(
    resultados_kurtosis,
    columns=["variable", "kurtosis_original", "kurtosis_log"]
)

# Ordenar por mayor curtosis original para identificar colas pesadas
tabla_kurtosis = tabla_kurtosis.sort_values(by="kurtosis_original", ascending=False)

# Configuración para mostrar toda la tabla
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

print(tabla_kurtosis.to_string(index=False))

#-----------------------------------------------------------
#NUEVAS VARIABLES
#-----------------------------------------------------------
print("\n" + "="*60)
print("Logaritmo de amt")
try:
    data["amt_log"] = np.log1p(data["amt"])
    print("Primeros 3 valores de amt_log:")
    print(data["amt_log"].head(3))
except Exception as e:
    print(f"Error al calcular amt_log: {e}")

print("\n" + "="*60)
print("Logaritmo de city_pop")
try:
    data["city_pop_log"] = np.log1p(data["city_pop"])
    print("Primeros 3 valores:")
    print(data["city_pop_log"].head(3))
except Exception as e:
    print(f"Error al calcular city_pop_log: {e}")

print("\n" + "="*60)
print("Logaritmo de velocidad (online y local)")
try:
    data["velocidad_log_local"] = np.log1p(data["velocidad_local"])
    data["velocidad_log_internet"] = np.log1p(data["velocidad_internet"])
    print("Ultimos 3 valores:")
    print(
        data[
            ["velocidad_log_local", "velocidad_log_internet"]
        ].tail(3)
    )
except Exception as e:
    print(f"Error al calcular velocidad_log: {e}")

print("\n" + "="*60)
print("Logaritmo de delta tiempo (online y local)")
try:
    data["delta_tiempo_log_local"] = np.log1p(data["delta_tiempo_local"])
    data["delta_tiempo_log_internet"] = np.log1p(data["delta_tiempo_internet"])
    print("Ultimos 3 valores:")
    print(
        data[
            [
                "delta_tiempo_log_local",
                "delta_tiempo_log_internet"
            ]
        ].tail(3)
    )
except Exception as e:
    print(f"Error al calcular delta_tiempo_log: {e}")
