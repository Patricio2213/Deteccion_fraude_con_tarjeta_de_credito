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
def boxplots(df, columnas_num, target="is_fraud"):
    print(f"Generando {len(columnas_num)} Boxplots...\n")

    for col in columnas_num:
        plt.figure(figsize=(8, 6))

        sns.boxplot(
            data=df,
            x=target,
            y=col,
            palette=["steelblue", "darkorange"],
            showfliers=True  # CRÍTICO: True para ver los outliers (el fraude)
        )

        plt.title(f"Análisis de Extremos: {col} vs Fraude")
        plt.xlabel("Es Fraude (0 = No, 1 = Sí)")
        plt.ylabel(col)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

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




