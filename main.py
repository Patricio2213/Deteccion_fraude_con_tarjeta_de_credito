#Carga de funciones
import gc
import statsmodels.api as sm
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
cols_a_eliminar = ['Unnamed: 0', 'cc_num', 'unix_time',"is_fraud","zip"]
cols_a_eliminar2 = ["trans_num","trans_date_trans_time"]

# Crear una lista con el nombre de las variables categóricas
cat_columns = data.select_dtypes(include=['object', 'string']).drop(columns=cols_a_eliminar2,errors="ignore").columns

# Creamos la lista de numéricas excluyendo las de arriba
num_columns = data.select_dtypes(include=['number']).drop(columns=cols_a_eliminar, errors='ignore').columns

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

del data_test
del data_train

#forzar limpieza
gc.collect()
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

#--------------------------------------------------------
#MODELOS
#--------------------------------------------------------

"""
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])

# 2. DEFINICIÓN DE VENTANAS TEMPORALES (Validación Temporal)
fecha_entreno_inicio = '2019-01-01 00:00:18'
fecha_entreno_fin    = '2020-06-21 12:13:37'
fecha_prueba_inicio  = '2020-06-21 12:14:25'
fecha_prueba_fin     = '2020-12-31 23:59:34'

train_df = data[(data['trans_date_trans_time'] >= fecha_entreno_inicio) &
              (data['trans_date_trans_time'] <= fecha_entreno_fin)].copy()
test_df  = data[(data['trans_date_trans_time'] >= fecha_prueba_inicio) &
              (data['trans_date_trans_time'] <= fecha_prueba_fin)].copy()

print(f"📊 Entrenamiento: {train_df.shape[0]} registros")
print(f"📊 Prueba: {test_df.shape[0]} registros")

# --- 3. INGENIERÍA DE VARIABLES TEMPORALES ---
for df_temp in [train_df, test_df]:
    df_temp['hour'] = df_temp['trans_date_trans_time'].dt.hour
    df_temp['day_of_week'] = df_temp['trans_date_trans_time'].dt.dayofweek
    df_temp['is_weekend'] = df_temp['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# 4. SELECCIÓN DE VARIABLES
target = 'is_fraud'
columnas_drop = [target, "trans_date_trans_time","street","first","last","merchant","city","dob","persona_id"]

X_train = train_df.drop(columns=[col for col in columnas_drop if col in train_df.columns])
y_train = train_df[target]
X_test  = test_df.drop(columns=[col for col in columnas_drop if col in test_df.columns])
y_test  = test_df[target]

cat_columns = [col for col in cat_columns if col in X_train.columns]
# --- 5. PREPROCESAMIENTO (ColumnTransformer) ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_columns)
    ])

X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# Convertir a array denso si el encoder devuelve matriz dispersa (necesario para DL)
if hasattr(X_train_scaled, "toarray"):
    X_train_scaled = X_train_scaled.toarray()
    X_test_scaled = X_test_scaled.toarray()

# --- 6. ENTRENAMIENTO MODELOS RÁPIDOS (Sklearn/XGBoost) ---
print("\n--- Entrenando Modelos de Respuesta Rápida ---")
# --- 7. MODELOS SUPERVISADOS (ESTADÍSTICA Y ML) ---
print("\n--- Entrenando Modelos Supervisados ---")

# 1. LIMPIEZA INICIAL
# Asegúrate de haber hecho 'del df1, df2' antes de llegar aquí
gc.collect()

# A. REGRESIÓN LOGÍSTICA (SUMMARY ESTADÍSTICO)
print("⏳ Ajustando Logit (Sin duplicación innecesaria)...")

try:
    # Creamos la columna de unos
    ones_train = np.ones((X_train_scaled.shape[0], 1), dtype=X_train_scaled.dtype)

    # Unimos para el summary. Esta es la ÚNICA copia permitida para Statsmodels.
    X_train_stat = np.ascontiguousarray(np.hstack([ones_train, X_train_scaled]))

    # Liberamos el vector de unos inmediatamente
    del ones_train
    gc.collect()

    # Ajustamos con L-BFGS (Crucial para no generar una matriz Hessiana gigante en RAM)
    logit_mod = sm.Logit(y_train, X_train_stat)
    logit_res = logit_mod.fit(method='lbfgs', maxiter=100, disp=0)

    print("\n=== SUMMARY DE REGRESIÓN LOGÍSTICA ===")
    print(logit_res.summary())

    # Evaluación de la Regresión
    ones_test = np.ones((X_test_scaled.shape[0], 1), dtype=X_test_scaled.dtype)
    X_test_stat = np.hstack([ones_test, X_test_scaled])
    del ones_test

    y_prob_logit = logit_res.predict(X_test_stat)
    evaluar_modelo("Regresión Logística", y_test, y_prob_logit)

    # LIMPIEZA TOTAL de las matrices de la regresión antes de seguir
    del X_train_stat, X_test_stat, logit_mod
    gc.collect()

except MemoryError:
    print("❌ Error de RAM: El sistema no soporta la matriz de 1.3M en float64.")
    print("💡 Intenta: X_train_stat = sm.add_constant(X_train_scaled[:500000]) para el summary.")

# B. OTROS SUPERVISADOS (XGBoost)
# XGBoost es mucho más eficiente y usará la matriz X_train_scaled original
print(f"\n🚀 Entrenando XGBoost...")
xgb_model = get_xgboost()
xgb_model.fit(X_train_scaled, y_train)
y_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
evaluar_modelo("XGBoost", y_test, y_prob_xgb)

# --- 8. MODELOS NO SUPERVISADOS ---
print("\n--- Entrenando Familias No Supervisadas ---")
iso_forest = get_isolation_forest()
iso_forest.fit(X_train_scaled)

y_prob_iso = -iso_forest.decision_function(X_test_scaled)
# Normalización rápida
y_prob_iso = (y_prob_iso - y_prob_iso.min()) / (y_prob_iso.max() - y_prob_iso.min() + 1e-9)
evaluar_modelo("Isolation Forest", y_test, y_prob_iso)

# --- 9. DEEP LEARNING (MLP & AUTOENCODER) ---
print("\n--- Entrenando Deep Learning ---")
# PyTorch requiere float32 para ser eficiente.
# Si le pasas float64, él creará una copia interna, duplicando tu RAM.
# Para evitarlo, convertimos AQUÍ y borramos la original si es necesario.

X_train_tensor = torch.from_numpy(X_train_scaled.astype('float32'))
X_test_tensor = torch.from_numpy(X_test_scaled.astype('float32'))
y_train_tensor = torch.FloatTensor(y_train.values.copy()).view(-1, 1)
input_dim = X_train_tensor.shape[1]

# A. MLP
print(f"🚀 Entrenando MLP...")
mlp_model = MLP(input_dim)
criterion_mlp = torch.nn.BCELoss()
optimizer_mlp = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

for epoch in range(50):
    mlp_model.train()
    optimizer_mlp.zero_grad()
    outputs = mlp_model(X_train_tensor)
    loss = criterion_mlp(outputs, y_train_tensor)
    loss.backward()
    optimizer_mlp.step()
    if (epoch + 1) % 10 == 0:
        print(f"MLP - Época [{epoch + 1}/50], Loss: {loss.item():.4f}")

mlp_model.eval()
with torch.no_grad():
    y_prob_mlp = mlp_model(X_test_tensor).numpy().flatten()
    evaluar_modelo("MLP (Red Neuronal)", y_test, y_prob_mlp)

# B. AUTOENCODER
print(f"🚀 Entrenando Autoencoder...")
ae_model = Autoencoder(input_dim)
criterion_ae = torch.nn.MSELoss()
optimizer_ae = torch.optim.Adam(ae_model.parameters(), lr=0.001)

for epoch in range(50):
    ae_model.train()
    optimizer_ae.zero_grad()
    reconstruction = ae_model(X_train_tensor)
    loss = criterion_ae(reconstruction, X_train_tensor)
    loss.backward()
    optimizer_ae.step()
    if (epoch + 1) % 10 == 0:
        print(f"Autoencoder - Época [{epoch + 1}/50], Loss: {loss.item():.4f}")

ae_model.eval()
with torch.no_grad():
    reconst_test = ae_model(X_test_tensor)
    mse_test = torch.mean((X_test_tensor - reconst_test) ** 2, dim=1).numpy()
    mse_test_norm = (mse_test - mse_test.min()) / (mse_test.max() - mse_test.min() + 1e-9)
    evaluar_modelo("Autoencoder", y_test, mse_test_norm)

print("\n--- Proceso finalizado con éxito ---")
"""