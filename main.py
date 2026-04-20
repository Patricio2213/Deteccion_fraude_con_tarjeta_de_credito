#Carga de funciones

from Subida_data import *
from procesamiento_bases import *
from EDA import *
from models import *

data_train=buscar_y_cargar("fraudTrain.csv")
data_test=buscar_y_cargar("fraudTest.csv")

# Crear una lista con el nombre de las variables categóricas
cat_columns = data_train.select_dtypes(include=['object', 'string']).columns


# Definir las que NO queremos considerando id y variables que no tenga sentido analizar
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


#-------------------------------------------------------
#  Calcular el mínimo y el máximo de las fechas
#-------------------------------------------------------
fecha_min_test = data_test["trans_date_trans_time"].min()
fecha_max_test = data_test["trans_date_trans_time"].max()

print("Periodo de data_train")
fecha_min_train = data_train["trans_date_trans_time"].min()
fecha_max_train = data_train["trans_date_trans_time"].max()

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
print("3. BALANCE DE CLASES DATA_TRAIN(is_fraud)")
print("="*60)
print(balance_clases(data_train)) #0.58%
print("\n" + "="*60)
print("\n" + "="*60)
print("BALANCE DE CLASES DATA_TEST(is_fraud)")
print("="*60)
print(balance_clases(data_test))
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
#for num_var in num_columns:
 #   graficar_boxplot_normal(data_train, num_var, target="is_fraud")

#for cat_var in cat_columns:
 #   make_barplot(data_train,cat_var,top=15) #univariado

#print("\n Matriz de correlación...")
#make_heat_map(data_train,num_columns)#multivariado

#graficar_temporalidad_fraude(data_train)
#fecha_min = data_train['trans_date_trans_time'].min()
#fecha_max = data_train['trans_date_trans_time'].max()

#print(f"Los datos comienzan el: {fecha_min}")
#print(f"Los datos terminan el: {fecha_max}")
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
#stats_vel_original = data_train.groupby('is_fraud')['velocidad'].describe(percentiles=[.25, .5, .75])

#  Cálculo de los componentes del Boxplot
#stats_vel_original['IQR'] = stats_vel_original['75%'] - stats_vel_original['25%']

# Bigote Superior: Donde estadísticamente empiezan los outliers
#stats_vel_original['Bigote_Superior'] = stats_vel_original['75%'] + (1.5 * stats_vel_original['IQR'])

# Mostrar la tabla
#print("="*65)
#print("TABLA ESTADÍSTICA")
#print("="*65)
#print(stats_vel_original.T)

#tabla_a, tabla_b = generar_tablas_tesis(data_train, num_columns)

#print("\n" + "="*80)
#print("TABLA A: ESTADÍSTICAS DESCRIPTIVAS POR GRUPO")
#print("="*80)
#print(tabla_a.to_string(index=False))

#print("\n" + "="*80)
#print("TABLA B: MÉTRICAS DE COMPARACIÓN Y SEPARACIÓN (FRAUDE VS LEGÍTIMO)")
#print("="*80)
#print(tabla_b.to_string(index=False))


# Veamos las categorias de comercio

# Categorías distintas ordenadas alfabéticamente
#categorias_unicas = sorted(data_train["category"].dropna().unique())

#print("Listado completo de categorías:")
#for cat in categorias_unicas:
 #   print(cat)

#print("\nCantidad total de categorías distintas:", len(categorias_unicas))

# VISUALIZAR UBICACIÓN DE COMERCIOS diferenciando net del resto

# Tomo una muestra para no saturar el gráfico
#df_plot = data_train.sample(n=1000000, random_state=50).copy()

# Definir categorías online
#categorias_net = ["grocery_net", "misc_net", "shopping_net"]

#data_train["tipo"] = np.where(
 #   data_train["category"].isin(categorias_net),
  #  "net",
   # "resto"
#)

# Gráfico
#plt.figure(figsize=(10, 6))

#sns.scatterplot(
 #   data=data_train,
  #  x="merch_long",
   # y="merch_lat",
    #hue="tipo",
    #alpha=0.5
#)

#plt.title("Ubicación de comercios (net vs resto)")
#plt.xlabel("Longitud")
#plt.ylabel("Latitud")
#plt.legend()
#plt.grid(alpha=0.3)

#plt.show()


# Creo una variable: ES ONLINE (1) VS FÍSICO (0)

categorias_net = ["grocery_net", "misc_net", "shopping_net"]

data_train["es_online"] = data_train["category"].isin(categorias_net).astype(int)

print("Conteo de compras online vs físicas:")
print(data_train["es_online"].value_counts())

print("\nPorcentaje:")
print(data_train["es_online"].value_counts(normalize=True) * 100)

grafico_tasa_por_variable(data_train, "es_online")

#Veamos que variables podrían necesitar log

# DIAGNÓSTICO DE VARIABLES NUMÉRICAS:
# VER SI CONVIENE USAR LOG, ZSCORE O DEJARLA IGUAL

#def diagnostico_transformaciones(df, columnas_numericas):
 #   resultados = []

  #  for col in columnas_numericas:
   #     serie = df[col].dropna()

        # Si la variable no tiene datos suficientes, la marco para revisión
    #    if len(serie) < 5:
     #       resultados.append({
      #          "Variable": col,
       #         "Min": np.nan,
        #        "Q1": np.nan,
         #       "Mediana": np.nan,
          #      "Q3": np.nan,
           #     "Max": np.nan,
            #    "Skew": np.nan,
             #   "%_Ceros": np.nan,
              #  "%_Negativos": np.nan,
               # "%_Outliers_IQR": np.nan,
                #"Sugerencia": "revisar",
                #"Motivo": "muy pocos datos"
            #})
            #continue

        # Estadísticos básicos
        #q1 = serie.quantile(0.25)
        #mediana = serie.median()
        #q3 = serie.quantile(0.75)
        #iqr = q3 - q1
        #min_val = serie.min()
        #max_val = serie.max()
        #skew_val = serie.skew()

        # Porcentaje de ceros y negativos
        #pct_ceros = (serie == 0).mean() * 100
        #pct_negativos = (serie < 0).mean() * 100

        # Outliers usando criterio IQR
        #lim_inf = q1 - 1.5 * iqr
        #lim_sup = q3 + 1.5 * iqr
        #pct_outliers = ((serie < lim_inf) | (serie > lim_sup)).mean() * 100


        # REGLAS METODOLÓGICAS PARA SUGERIR TRANSFORMACIÓN


        # Caso 1: si tiene negativos, no conviene aplicar log directo
        #if pct_negativos > 0:
         #   sugerencia = "zscore o nada"
          #  motivo = "tiene valores negativos, log no aplica directo"

        # Caso 2: si está muy sesgada a la derecha y no tiene negativos
        #elif skew_val > 2:
         #   sugerencia = "log"
          #  motivo = "alta asimetría positiva"

        # Caso 3: si está moderadamente sesgada y con bastantes outliers
        #elif skew_val > 1 and pct_outliers > 5:
         #   sugerencia = "log"
          #  motivo = "sesgo positivo y presencia de outliers"

        # Caso 4: si no está tan sesgada, pero tiene escala rara o outliers
        #elif abs(skew_val) <= 1 and pct_outliers > 5:
         #   sugerencia = "zscore"
          #  motivo = "distribución razonable, pero con outliers/escala"

        # Caso 5: distribución bastante estable
        #else:
         #   sugerencia = "nada o zscore"
          #  motivo = "distribución relativamente estable"

        #resultados.append({
         #   "Variable": col,
          #  "Min": round(min_val, 3),
           # "Q1": round(q1, 3),
            #"Mediana": round(mediana, 3),
            #"Q3": round(q3, 3),
            #"Max": round(max_val, 3),
            #"Skew": round(skew_val, 3),
            #"%_Ceros": round(pct_ceros, 2),
            #"%_Negativos": round(pct_negativos, 2),
            #"%_Outliers_IQR": round(pct_outliers, 2),
            #"Sugerencia": sugerencia,
            #"Motivo": motivo
        #})

    #tabla_diag = pd.DataFrame(resultados).sort_values(
     #   by=["Sugerencia", "Skew"],
      #  ascending=[True, False]
    #)

    #return tabla_diag



# EJECUCIÓN DEL DIAGNÓSTICO (veamos qué columnas podrían necesitar log o algun procesamiento extra)
# EXCLUYO LA VARIABLE OBJETIVO Y OTRAS QUE NO QUIERO EVALUAR


#columnas_revisar = [
 #   col for col in data_train.select_dtypes(include=["number"]).columns
  #  if col not in ["is_fraud"]
#]

#tabla_diagnostico = diagnostico_transformaciones(data_train, columnas_revisar)

#print("\n" + "=" * 120)
#print("DIAGNÓSTICO DE TRANSFORMACIONES")
#print("=" * 120)
#print(tabla_diagnostico.to_string(index=False))

#MODELOS

