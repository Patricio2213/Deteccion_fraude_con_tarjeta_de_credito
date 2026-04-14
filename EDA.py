

#DENSIDAD
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import itertools

plt.style.use('fivethirtyeight')

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

            # Guardar el gráfico con el nombre de la columna
        plt.savefig(f"densidad_{col}.pdf", bbox_inches='tight', dpi=300)
        plt.tight_layout()
        plt.show()
 #EDA PROFESOR

def make_barplot(dataframe, cat_var,top):
    # Calcular porcentajes
    percentages = dataframe[cat_var].value_counts(normalize=True).head(top) * 100

    # Tamaño de la figura
    plt.figure(figsize=(16, 9))

    # Crear el diagrama de barras
    my_fig = percentages.plot(kind='bar')

    # Añadir porcentajes a las barras
    for i, valor in enumerate(percentages):
        my_fig.text(i, valor + 0.2, f"{valor:.2f}%", fontsize=16, va='center', ha='center',
                    color='black', weight='bold')

    # Añadir títulos a los ejes
    plt.xlabel(cat_var)
    plt.ylabel('Percentage (%)')

    # Rotar las leyendas del eje x
    plt.xticks(rotation=45,ha="right")
    plt.ylim(0, percentages.max() + 5)

    # Controlar las lineas horizontales y verticales
    my_fig.grid(axis='x', visible=False)
    my_fig.grid(axis='y', visible=True)
    # Guardar el gráfico usando el nombre de la variable analizada
    plt.savefig(f"barplot_{cat_var}.pdf", bbox_inches='tight', dpi=300)
    # Mostrar la figura
    plt.tight_layout()
    plt.show()

def make_boxplot(dataframe, num_var, unit=''):
    # Tamaño de la figura
    plt.figure(figsize=(16, 9))

    # Crear el boxplot
    sns.boxplot(dataframe, x=num_var)

    # Etiquetas de los ejes
    plt.xlabel(num_var)
    plt.ylabel('Values')

    # Calcular estadísticas descriptivas
    mean_val = dataframe[num_var].mean()
    std_val = dataframe[num_var].std()
    median_val = dataframe[num_var].median()
    min_val = dataframe[num_var].min()
    max_val = dataframe[num_var].max()
    skew_val = dataframe[num_var].skew()

    # Guardar las estadísticas en una tupla
    stats_text = (
        f"Mean: {mean_val:.2f} {unit}\n"
        f"Median: {median_val:.2f} {unit}\n"
        f"Std: {std_val:.2f} {unit}\n"
        f"Min: {min_val:.2f} {unit}\n"
        f"Max: {max_val:.2f} {unit}\n"
        f"Skew: {skew_val:.2f}"
    )

    # Colocar las estadísticas en la figura
    plt.text(
        0.95, 0.95, stats_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        multialignment='right',
        fontfamily='monospace',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )
    # Guardar el gráfico usando el nombre de la variable analizada
    plt.savefig(f"boxplot_{num_var}.pdf", bbox_inches='tight', dpi=300)
    # Mostrar la figura
    plt.tight_layout()
    plt.show()

def make_heat_map(dataframe, num_vars):
    # Definir tamaño de la figura
    plt.figure(figsize=(16, 9))

    # Calcular la matriz de correlación
    corr_matrix = dataframe[num_vars].corr()

    # Crear una máscara para esconder mitad de la matriz
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Crear el heatmap
    sns.set_theme(font_scale=1.2)
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
    plt.savefig("coor_matrix", bbox_inches='tight')
    # Mostrar la figura
    plt.tight_layout()
    plt.show()


def make_scatter_plot(dataframe, num_vars):
    # Tamaño de la figura
    plt.figure(figsize=(16, 9))

    # Generar la matriz de scatter plot
    g = sns.pairplot(dataframe[num_vars], corner=False, diag_kind='kde')

    # Ajustar las etiquetas del eje y
    for ax in g.axes.flatten():
        if ax is not None:
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_ha('right')
            ax.yaxis.labelpad = 13

    # Ajustar la figura para evitar traslape
    g.figure.tight_layout()
    plt.show()

def make_grouped_boxplots(dataframe, num_vars, cat_vars, type_plot='boxplot', unit=''):

    for num_var, cat_var in itertools.product(num_vars, cat_vars):
        # Tamaño de la figura
        plt.figure(figsize=(16, 9))

        # Boxplot o Violinplot
        if type_plot == 'boxplot':
            ax = sns.boxplot(x=cat_var, y=num_var, data=dataframe)
        else:
            ax = sns.violinplot(x=cat_var, y=num_var, data=dataframe, box=None)

        # Etiquetas de los ejes
        plt.xlabel(cat_var)
        plt.ylabel(f"{num_var} ({unit})" if unit else num_var)

        # Calcular las estadísticas por grupo
        grouped_stats = dataframe.groupby(cat_var)[num_var].agg(
            mean='mean',
            median='median',
            std='std',
            min='min',
            max='max',
            skew='skew'
        )

        # Formatear las estadísticas verticalmente
        stats_lines = [""]
        for cat, row in grouped_stats.iterrows():
            stats_lines.append(f"{cat}:")
            stats_lines.append(f"  Mean: {row['mean']:.2f} {unit}")
            stats_lines.append(f"  Median: {row['median']:.2f} {unit}")
            stats_lines.append(f"  Std: {row['std']:.2f} {unit}")
            stats_lines.append(f"  Min: {row['min']:.2f} {unit}")
            stats_lines.append(f"  Max: {row['max']:.2f} {unit}")
            stats_lines.append(f"  Skew: {row['skew']:.2f}")
            stats_lines.append("")

        stats_text = "\n".join(stats_lines)

        # Graficar las estadísticas afuera de la figura
        plt.gcf().subplots_adjust(right=0.7)
        plt.text(
            1.01, 1, stats_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='left',
            multialignment='left',
            fontfamily='monospace',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

        # Mostrar la figura
        plt.tight_layout()
        plt.show()


def make_stacked_barplots(dataframe, cat_vars, top=10):
    # Forzamos que 'is_fraud' esté siempre presente para la comparación
    target = "is_fraud"

    # Quitamos 'is_fraud' de cat_vars si es que viene adentro, para no compararlo consigo mismo
    variables_a_graficar = [v for v in cat_vars if v != target]

    for var1 in variables_a_graficar:
        # 1. Identificamos las categorías más frecuentes del eje X
        top_categories = dataframe[var1].value_counts().head(top).index

        # 2. Filtramos el dataframe
        df_filtered = dataframe[dataframe[var1].isin(top_categories)]

        # 3. Tabla de contingencia (Cruce Variable x Fraude)
        # Usamos df_filtered[target] para que siempre use la columna de fraude
        crosstab = pd.crosstab(df_filtered[var1], df_filtered[target], normalize='index') * 100

        # Reordenamos por frecuencia
        crosstab = crosstab.reindex(top_categories)

        # --- Gráfico ---
        plt.figure(figsize=(16, 9))
        ax = crosstab.plot(kind='bar', stacked=True, ax=plt.gca(), color=['steelblue', 'darkorange'])

        plt.title(f"Relación: {var1} (Top {top}) vs {target}", fontsize=16, pad=20)
        plt.ylabel('Percentage (%)')
        plt.xlabel(var1)

        # Añadir el texto dentro de las barras
        for i, row in enumerate(crosstab.values):
            cumulative = 0
            for j, value in enumerate(row):
                if value > 1:
                    ax.text(
                        i,
                        cumulative + value / 2,
                        f"{value:.1f}%",
                        ha='center', va='center', color='white', fontweight='bold', fontsize=10
                    )
                cumulative += value

        plt.legend(title=target, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', visible=True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
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

def graficar_temporalidad_fraude(df):
    df_temp = df.copy()
        # Aseguramos formato datetime
    df_temp["trans_date_trans_time"] = pd.to_datetime(df_temp["trans_date_trans_time"])

        # Extraemos dimensiones micro
    df_temp["año"] = df_temp["trans_date_trans_time"].dt.year
    df_temp["mes"] = df_temp["trans_date_trans_time"].dt.month
    df_temp["hora"] = df_temp["trans_date_trans_time"].dt.hour

        # --- 1. COMPORTAMIENTO POR AÑO ---
    plt.figure(figsize=(12, 6))
      # Calculamos la tasa agrupando por año
    tasa_anual = df_temp.groupby("año")["is_fraud"].mean() * 100
    sns.barplot(x=tasa_anual.index, y=tasa_anual.values, color="blue")
    plt.title("Evolución Anual: ¿Está aumentando el fraude año a año?", fontsize=14)
    plt.ylabel("% de Fraude")
    plt.savefig("temp_1_anual.pdf", bbox_inches='tight')
    plt.tight_layout()
    plt.show()

        # --- 2. COMPORTAMIENTO POR MES (ESTACIONALIDAD) ---
    plt.figure(figsize=(12, 6))
        # Calculamos la tasa agrupando por mes
    tasa_mensual = df_temp.groupby("mes")["is_fraud"].mean() * 100
    sns.lineplot(x=tasa_mensual.index, y=tasa_mensual.values, marker="s", color="darkorange", linewidth=3)
    plt.title("Estacionalidad Mensual: ¿Hay meses más peligrosos?", fontsize=14)
    plt.xticks(range(1, 13), ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
    plt.ylabel("% de Fraude")
    plt.grid(alpha=0.3)
    plt.savefig("temp_2_mensual.pdf", bbox_inches='tight')
    plt.tight_layout()
    plt.show()

        # --- 3. EL "RELOJ" COMPARATIVO (AÑO VS HORA) ---
    plt.figure(figsize=(12, 6))
        # Aquí comparamos cómo cambia el reloj según el año
    tasa_hora_año = df_temp.groupby(["hora", "año"])["is_fraud"].mean().reset_index()
    tasa_hora_año["is_fraud"] *= 100

    sns.lineplot(data=tasa_hora_año, x="hora", y="is_fraud", hue="año", marker="o")
    plt.title("Reloj del Criminal: Comparativa por Años", fontsize=14)
    plt.xlabel("Hora del Día")
    plt.ylabel("% de Fraude")
    plt.xticks(range(0, 24))
    plt.legend(title="Año")
    plt.savefig("temp_3_comparativo.pdf", bbox_inches='tight')
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



