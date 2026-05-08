#Carga de funciones
import gc
from Subida_data import *
from procesamiento_bases import *
from EDA import *
from models import *
import statsmodels.api as sm

#Carga de bases
data_train=buscar_y_cargar("fraudTrain.csv")
data_test=buscar_y_cargar("fraudTest.csv")
data=pd.concat([data_train, data_test], ignore_index=True)
#print(data.head())

# Definir las que NO queremos considerando id y variables que no tengan sentido utilizar en base al EDA
cols_a_eliminar = ['Unnamed: 0', 'cc_num', 'unix_time',"is_fraud","zip"]
cols_a_eliminar2 = ["trans_num","trans_date_trans_time"]

# Crear una lista con el nombre de las variables categóricas
cat_columns = data.select_dtypes(include=['object', 'string']).drop(columns=cols_a_eliminar2,errors="ignore").columns

# Creamos la lista de numéricas excluyendo las de arriba
num_columns = data.select_dtypes(include=['number']).drop(columns=cols_a_eliminar, errors='ignore').columns

"""
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
"""
del data_test
del data_train

#forzar limpieza
gc.collect()
"""
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

"""
# Creación de nuevas variables

data = distancia_entre_comercios(data)
data = calcular_velocidad(data)
data = distancia_cliente_comercio(data)
"""
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
"""
#TABLAS SIN UNSO ACTUALMENTE PQ NO SE HAN AÑADIDO LAS NUEVAS

# Definir categorías online
categorias_net = ["grocery_net", "misc_net", "shopping_net"]

 #Creo una variable: ES ONLINE (1) VS FÍSICO (0)

data["es_online"] = data["category"].isin(categorias_net).astype(int)

#print("Conteo de compras online vs físicas:")
#print(data["es_online"].value_counts())

#print("\nPorcentaje:")
#print(data["es_online"].value_counts(normalize=True) * 100)

#grafico_tasa_por_variable(data, "es_online")
"""
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
"""
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
#--------------------------------------------------------
#EDA NUEVAS VARIABLES
#--------------------------------------------------------
num_nuevas=["delta_tiempo_log_local","delta_tiempo_log_internet","city_pop_log","amt_log","velocidad_log_local","velocidad_log_internet","distancia_local","distancia_internet","d_cliente_comercio_int","d_cliente_comercio_loc","edad","tasa_categoria","velocidad_internet","velocidad_local","delta_tiempo_local","delta_tiempo_internet"]
cat_nueva=["es_nuevo","es_online","is_first_internet","is_first_local"]
"""
graficar_densidad(data,num_nuevas)
for num_var in num_nuevas:
    make_boxplot(data,num_var)
"""



#ANÁLISIS MULTIVARIADO

#boxplots_con_tabla(data, num_nuevas,target="is_fraud")

# Actualizar la lista con el nombre de las variables categóricas
cat_columns = data.select_dtypes(include=['object', 'string']).drop(columns=cols_a_eliminar2,errors="ignore").columns

# Actualizar la lista de numéricas excluyendo las de arriba
num_columns = data.select_dtypes(include=['number']).drop(columns=cols_a_eliminar, errors='ignore').columns
#--------------------------------------------------------
#MODELOS
#--------------------------------------------------------


# --- 1. MUESTREO Y PREPARACIÓN ---
# ---  FILTRADO TEMPORAL Y MUESTREO ÚNICO (Al inicio de todo) ---
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])

fecha_entreno_inicio = '2019-01-01 00:00:18'
fecha_entreno_fin    = '2020-06-21 12:13:37'
fecha_prueba_inicio  = '2020-06-21 12:14:25'
fecha_prueba_fin     = '2020-12-31 23:59:34'

# Filtramos los universos completos por fecha una sola vez
full_train = data[(data['trans_date_trans_time'] >= fecha_entreno_inicio) &
                 (data['trans_date_trans_time'] <= fecha_entreno_fin)]
full_test = data[(data['trans_date_trans_time'] >= fecha_prueba_inicio) &
                (data['trans_date_trans_time'] <= fecha_prueba_fin)]

# MUESTREO ÚNICO: Aquí definimos el set que usarán todos los modelos
n_train = 160000
n_test  = 40000

train_df = full_train.sample(n=min(n_train, len(full_train)), random_state=42).copy()
test_df  = full_test.sample(n=min(n_test, len(full_test)), random_state=42).copy()

# Liberamos memoria de los objetos grandes
del full_train, full_test, data
import gc
gc.collect()

# --- 2. INGENIERÍA DE VARIABLES TEMPORALES ---
for df_temp in [train_df, test_df]:
    df_temp['hour'] = df_temp['trans_date_trans_time'].dt.hour
    df_temp['day_of_week'] = df_temp['trans_date_trans_time'].dt.dayofweek
    df_temp['is_weekend'] = (df_temp['day_of_week'] >= 5).astype(int)

# --- 3. SELECCIÓN DE VARIABLES Y PREPROCESAMIENTO ---
target = 'is_fraud'
columnas_drop = [target, "trans_date_trans_time", "street", "first", "last",
                 "merchant", "city", "dob", "persona_id","lat","long","merch_lat","merch_long","job","amt","city_pop","velocidad"]

# Creamos los sets de datos base
X_train_raw = train_df.drop(columns=[col for col in columnas_drop if col in train_df.columns])
X_test_raw  = test_df.drop(columns=[col for col in columnas_drop if col in test_df.columns])
y_train = train_df[target]
y_test  = test_df[target]

num_columns = [col for col in num_columns if col in X_train_raw.columns]
cat_columns = [col for col in cat_columns if col in X_train_raw.columns]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_columns),
        ('cat', OneHotEncoder(drop='first',handle_unknown='ignore', sparse_output=False), cat_columns)
    ])

X_train_scaled = preprocessor.fit_transform(X_train_raw)
X_test_scaled = preprocessor.transform(X_test_raw)

# --- 4. MODELOS SUPERVISADOS TRADICIONALES ---
print("\n--- Entrenando Modelos Supervisados ---")

# A. REGRESIÓN LOGÍSTICA (Statsmodels)
print("⏳ Ajustando Logit...")

# 1. Recuperar nombres de columnas
nombres_columnas = preprocessor.get_feature_names_out()

# 2. Crear DataFrames y REINICIAR ÍNDICES de las etiquetas
X_train_final = pd.DataFrame(X_train_scaled, columns=nombres_columnas)
X_test_final = pd.DataFrame(X_test_scaled, columns=nombres_columnas)

# Reseteamos y_train e y_test
y_train_reset = y_train.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

# 3. Agregar constante
X_train_stat = sm.add_constant(X_train_final)
X_test_stat = sm.add_constant(X_test_final)

# 4. Asegurar que las columnas de Test sean IDENTICAS a las de Train
# Si falta una columna en Test que estaba en Train, la crea con ceros
X_test_stat = X_test_stat.reindex(columns=X_train_stat.columns, fill_value=0)

print(f"✅ Columnas Train: {X_train_stat.shape[1]} | Columnas Test: {X_test_stat.shape[1]}")

# 5. Ajustar modelo
logit_mod = sm.Logit(y_train_reset, X_train_stat)

try:
    # Usamos bfgs que es más tolerante a la cuasi-separación
    logit_res = logit_mod.fit(method='bfgs', maxiter=100, disp=0)
    print("\n=== SUMMARY DE REGRESIÓN LOGÍSTICA ===")
    print(logit_res.summary())

    y_prob_logit = logit_res.predict(X_test_stat)
    evaluar_modelo("Regresión Logística", y_test_reset, y_prob_logit, umbral=0.5)

except Exception as e:
    print(f"⚠️ Nota de Tesis: La Regresión Logística no convergió por inestabilidad numérica.")
    print(f"Detalle: {e}")

    # Ejemplo para una variable categórica
print(pd.crosstab(train_df['category'], train_df['is_fraud']))
import numpy as np
rank = np.linalg.matrix_rank(X_train_scaled)
print(f"Columnas totales: {X_train_scaled.shape[1]} | Rango: {rank}")

# B. XGBOOST con OPTIMIZACIÓN BAYESIANA
print(f"\n🚀 Iniciando Optimización Bayesiana para XGBoost (Rangos: Tayebi & El Kafhali)...")

# Crear el estudio de Optuna
study = optuna.create_study(direction='maximize')

# Ejecutamos la optimización (n_trials=20 es un buen inicio para no tardar horas)
# Pasamos X_train_scaled e y_train para que aprete de ellos
study.optimize(lambda trial: objective_xgboost(trial, X_train_scaled, y_train), n_trials=20)

print(f"✅ Mejores Hiperparámetros encontrados: {study.best_params}")

# Entrenamos el modelo final con los mejores parámetros encontrados
print(f"📊 Entrenando modelo final con parámetros óptimos...")
xgb_model_final = get_xgboost(study.best_params)
xgb_model_final.fit(X_train_scaled, y_train)

# Predicción y Evaluación
y_prob_xgb = xgb_model_final.predict_proba(X_test_scaled)[:, 1]
evaluar_modelo("XGBoost Optimizado", y_test, y_prob_xgb, umbral=0.5)

# --- 5. MODELOS NO SUPERVISADOS (Isolation Forest & LOF) ---
print("\n--- Entrenando Modelos de Anomalías ---")

# A. ISOLATION FOREST
iso_forest = get_isolation_forest()
iso_forest.fit(X_train_scaled)
y_prob_iso = -iso_forest.decision_function(X_test_scaled)
y_prob_iso = (y_prob_iso - y_prob_iso.min()) / (y_prob_iso.max() - y_prob_iso.min() + 1e-9)
evaluar_modelo("Isolation Forest", y_test, y_prob_iso,umbral=0.5)

# B. LOCAL OUTLIER FACTOR (LOF) - METODOLOGÍA DE RANGOS (20-50)
print("⏳ Ejecutando LOF con metodología de rangos (Breunig et al.)...")

# Definimos el rango de vecinos a probar
rango_vecinos = [20, 30, 40, 50]
scores_acumulados = []

for k in rango_vecinos:
    print(f"   Procesando k={k}...")
    lof = get_lof(neighbors=k)
    lof.fit(X_train_scaled)
    
    # Obtenemos el score para este k, usamos signo negativo porque score_samples devuelve valores negativos para anomalías
    current_scores = -lof.score_samples(X_test_scaled)
    scores_acumulados.append(current_scores)

# Tomar el máximo score LOF entre todos los k
# np.maximum.reduce compara los arrays y se queda con el valor más alto en cada posición

y_prob_lof_max = np.maximum.reduce(scores_acumulados)

# Normalización para el Benchmark
y_prob_lof_final = (y_prob_lof_max - y_prob_lof_max.min()) / (y_prob_lof_max.max() - y_prob_lof_max.min() + 1e-9)

evaluar_modelo("LOF (Rango 20-50)", y_test, y_prob_lof_final, umbral=0.5)

# --- 6. PREPARACIÓN PARA REDES NEURONALES (PyTorch) ---
X_train_tensor = torch.from_numpy(X_train_scaled.copy()).float()
X_test_tensor = torch.from_numpy(X_test_scaled.copy()).float()
y_train_tensor = torch.from_numpy(y_train.values.copy()).float().view(-1, 1)

input_dim = X_train_tensor.shape[1]
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

# --- 7. ENTRENAMIENTO MLP BALANCEADO ---
conteo_clases = y_train.value_counts()
peso_fraude = conteo_clases[0] / conteo_clases[1]
pos_weight = torch.tensor([peso_fraude]).float()

mlp_model = MLP(input_dim)
optimizer_mlp = torch.optim.Adam(mlp_model.parameters(), lr=0.001)
criterion_mlp = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

for epoch in range(50):
    mlp_model.train()
    for batch_X, batch_y in train_loader:
        optimizer_mlp.zero_grad()
        loss = criterion_mlp(mlp_model(batch_X), batch_y)
        loss.backward()
        optimizer_mlp.step()

mlp_model.eval()
with torch.no_grad():
    y_prob_mlp = torch.sigmoid(mlp_model(X_test_tensor)).numpy().flatten()
    evaluar_modelo("MLP Balanceado", y_test, y_prob_mlp, umbral=0.5)

# --- 8. ENTRENAMIENTO AUTOENCODER ---
X_train_normal = X_train_tensor[y_train_tensor.flatten() == 0]
ae_train_loader = DataLoader(TensorDataset(X_train_normal), batch_size=32, shuffle=True)

ae_model = AutoencoderProgresivoVariable(input_dim, c1=16, c2=8, bottleneck=4)
optimizer_ae = torch.optim.Adam(ae_model.parameters(), lr=0.001)
criterion_ae = torch.nn.MSELoss()

for epoch in range(50):
    ae_model.train()
    for batch in ae_train_loader:
        batch_X = batch[0]
        optimizer_ae.zero_grad()
        loss = criterion_ae(ae_model(batch_X), batch_X)
        loss.backward()
        optimizer_ae.step()

ae_model.eval()
with torch.no_grad():
    reconst_test = ae_model(X_test_tensor)
    mse_test = torch.mean((X_test_tensor - reconst_test) ** 2, dim=1).numpy()
    # Sin umbral para usar el percentil dinámico
    evaluar_modelo("Autoencoder", y_test, mse_test)
