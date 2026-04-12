#VAR NUMERICAS Y CATEGORICAS
def separar_columnas(df, target="is_fraud", excluir=None):

    if excluir is None:
        excluir = []

    # Categóricas
    cat_columns= [col for col in df.select_dtypes(include=['object', 'string']).columns
        if col not in [target] + excluir
    ]

    # Numéricas (excluyendo target y otras), por si necesito separarlas
    num_columns= [
        col for col in df.select_dtypes(include=['number']).columns
        if col not in [target] + excluir
    ]

    return num_columns, cat_columns

#DENSIDAD
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

def graficar_densidad(df, columnas_num, target="is_fraud", auto_zoom=True):

    print(f"Iniciando generación de {len(columnas_num)} gráficos de densidad...\n")
    print("=" * 60)

    for col in columnas_num:
        plt.figure(figsize=(10, 6))


        sns.kdeplot(
            data=df,
            x=col,
            hue=target,
            common_norm=False,
            fill=True,
            alpha=0.5,
            palette=["steelblue", "darkorange"],
            linewidth=2
        )

        plt.xlabel(col)
        plt.ylabel("Densidad (Proporción)")
        plt.title(f"Distribución de Densidad: {col}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.grid(axis='x', visible=False)

        limite_inferior = df[col].min()
        limite_superior = df[col].quantile(0.99)

        #Aplicamos zoom si el percentil 99 es mayor al mínimo (evita errores con constantes)
        if limite_superior > limite_inferior:
            plt.xlim(limite_inferior, limite_superior)

        plt.tight_layout()
        plt.show()

#BOXPLOT
def boxplots_con_tabla(df, columnas_num, target="is_fraud"):
    print(f"Generando {len(columnas_num)} Boxplots con sus tablas estadísticas...\n")

    for col in columnas_num:
        #se genera el gráfico
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=df,
            x=target,
            y=col,
            palette=["steelblue", "darkorange"],
            showfliers=True
        )
        plt.title(f"Análisis de Extremos: {col} vs Fraude")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        # Cálculo de los componentes del Boxplot
        # describe para obtener 25%, 50% (mediana) y 75%
        stats = df.groupby(target)[col].describe(percentiles=[.25, .5, .75])

        # Cálculo de IQR
        stats['IQR'] = stats['75%'] - stats['25%']

        # Cálculo de Bigotes
        # El bigote superior es Q3 + 1.5 * IQR
        stats['Bigote_Superior'] = stats['75%'] + (1.5 * stats['IQR'])
        stats['Bigote_Inferior'] = stats['25%'] - (1.5 * stats['IQR'])

        # Identificación de Outliers (Valores fuera de los bigotes)
        def contar_outliers(group):
            b_sup = group.quantile(0.75) + 1.5 * (group.quantile(0.75) - group.quantile(0.25))
            return (group > b_sup).sum()

        stats['Cant_Outliers'] = df.groupby(target)[col].apply(contar_outliers)

        # Tabla para presentación
        print(f"ESTADÍSTICAS DE BOXPLOT PARA: {col}")
        # Transponemos (.T) para que sea más fácil de leer
        columnas_finales = ['count', 'min', '25%', '50%', '75%', 'max', 'IQR', 'Bigote_Inferior', 'Bigote_Superior',
                            'Cant_Outliers']
        print(stats[columnas_finales].T)
        print("-" * 60)

#Reloj fraude
def graficar_reloj_fraude(df):
    print("Generando el Reloj del Criminal...")

    # Aseguramos formato datetime y extraemos la hora
    df_temp = df.copy()
    df_temp["hora"] = pd.to_datetime(df_temp["trans_date_trans_time"]).dt.hour

    # Calculamos porcentaje de fraude por hora
    tasa_fraude_por_hora = df_temp.groupby("hora")["is_fraud"].mean() * 100

    plt.figure(figsize=(10, 5))
    sns.lineplot(
        x=tasa_fraude_por_hora.index,
        y=tasa_fraude_por_hora.values,
        marker="o",
        color="darkred",
        linewidth=2
    )

    plt.title("Tasa de Fraude según la Hora del Día")
    plt.xlabel("Hora del Día (0-23)")
    plt.ylabel("% de Transacciones que son Fraude")
    plt.xticks(range(0, 24))
    plt.grid(linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


    # Calculamos la correlación de Pearson
def graficar_correlacion(df, columnas_num):
    print("Generando Matriz de Correlación...")

    plt.figure(figsize=(10, 8))
    matriz_corr = df[columnas_num].corr()

    sns.heatmap(
        matriz_corr,
        annot=True,  # Muestra el número exacto
        fmt=".2f",  # 2 decimales
        cmap="coolwarm",  # Colores fríos y cálidos
        vmin=-1,
        vmax=1,
        linewidths=0.5
    )

    plt.title("Mapa de Calor: Multicolinealidad de Variables Numéricas")
    plt.tight_layout()
    plt.show()

def graficar_riesgo_porcategoria(df, columna, target= "is_fraud"):
    plt.figure(figsize=(12,6))
    riesgo= df.groupby(columna)[target].mean().sort_values(ascending=False) * 100
    sns.barplot(x=riesgo.values, y=riesgo.index, palette="Reds_r")
    plt.title(f"Tasa de riesgo de fraude por {columna}")
    plt.xlabel("Porcentaje de Fraude en esta categoría")
    plt.show()


def grafico_tasa_por_variable(df, columna, target="is_fraud"):

    plt.figure(figsize=(12, 6))

    # Calcular tasa por categoría
    tasa = df.groupby(columna)[target].mean().sort_values(ascending=False) * 100

    # Calcular promedio general
    tasa_promedio = df[target].mean() * 100

    # Asignar colores: rojo si ALTO riesgo, azul si BAJO
    colores = ["darkorange" if x > tasa_promedio else "steelblue" for x in tasa.values]

    # Crear bar plot
    ax = sns.barplot(x=tasa.index, y=tasa.values, palette=colores, edgecolor="black")

    # Línea de referencia (promedio)
    ax.axhline(y=tasa_promedio, color="red", linestyle="--", linewidth=2,
               label=f"Promedio ({tasa_promedio:.2f}%)")

    # Etiquetas
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(columna, fontweight='bold', fontsize=11)
    plt.ylabel("Tasa de Fraude (%)", fontweight='bold', fontsize=11)
    plt.title(f"Riesgo de Fraude por {columna}\n(Rojo: Mayor Riesgo | Azul: Menor Riesgo)",
              fontsize=12, fontweight='bold', pad=15)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def tabla_estadisticas_fraude(df, var_numericas, target="is_fraud"):

    resultados = []

    for col in var_numericas:
        legit = df[df[target] == 0][col]
        fraud = df[df[target] == 1][col]

        resultados.append({
            'Variable': col,
            'Legít_Media': f"{legit.mean():.2f}",
            'Legít_Std': f"{legit.std():.2f}",
            'Fraude_Media': f"{fraud.mean():.2f}",
            'Fraude_Std': f"{fraud.std():.2f}",
            'Diferencia': f"{(fraud.mean() - legit.mean()):.2f}",
            'Cambio_%': f"{((fraud.mean() - legit.mean()) / legit.mean() * 100):.1f}%"
        })

    tabla = pd.DataFrame(resultados)
    print("\n" + "=" * 100)
    print("TABLA: COMPARACIÓN DE ESTADÍSTICAS (FRAUDE vs LEGÍTIMO)")
    print("=" * 100)
    print(tabla.to_string(index=False))

    return tabla



def generar_tablas_tesis(df, var_numericas, target="is_fraud"):
    res_descriptivo = []
    res_comparativo = []

    for col in var_numericas:
        # Separar grupos según is_fraud [1, 2]
        legit = df[df[target] == 0][col].dropna()
        fraud = df[df[target] == 1][col].dropna()

        # --- TABLA A: DESCRIPTIVA (Mediana e IQR) ---
        med_l, iqr_l = legit.median(), legit.quantile(0.75) - legit.quantile(0.25)
        med_f, iqr_f = fraud.median(), fraud.quantile(0.75) - fraud.quantile(0.25)

        res_descriptivo.append({
            'Variable': col,
            'Mediana_Legít': f"{med_l:.2f}",
            'IQR_Legít': f"{iqr_l:.2f}",
            'Mediana_Fraude': f"{med_f:.2f}",
            'IQR_Fraude': f"{iqr_f:.2f}"
        })

        # --- TABLA B: COMPARATIVA (Métricas de Diferencia) ---
        # Cohen's d
        n1, n2 = len(legit), len(fraud)
        var1, var2 = legit.var(), fraud.var()
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohen_d = (fraud.mean() - legit.mean()) / pooled_std if pooled_std != 0 else 0

        # KS Statistic y Mann-Whitney U
        ks_stat, _ = stats.ks_2samp(legit, fraud)
        _, p_val = stats.mannwhitneyu(legit, fraud, alternative='two-sided')

        res_comparativo.append({
            'Variable': col,
            'Cohen_d': f"{cohen_d:.3f}",
            'KS_Stat': f"{ks_stat:.3f}",
            'MWU_p-val': f"{p_val:.4e}"
        })

    return pd.DataFrame(res_descriptivo), pd.DataFrame(res_comparativo)



