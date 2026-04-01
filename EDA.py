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

import matplotlib.pyplot as plt
import seaborn as sns


def graficar_densidad(df, columnas_num, target="is_fraud", auto_zoom=True):
    """
    Recorre una lista de variables y grafica la densidad comparativa automáticamente.
    """
    print(f"Iniciando generación de {len(columnas_num)} gráficos de densidad...\n")
    print("=" * 60)

    for col in columnas_num:
        plt.figure(figsize=(10, 6))

        # Magia de Seaborn: data=df y hue=target separan las clases solos
        # common_norm=False es vital para que no importe el desbalance de clases
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

        # Zoom automático inteligente para ignorar el 1% de valores más extremos
        if auto_zoom:
            limite_inferior = df[col].min()
            limite_superior = df[col].quantile(0.99)  # Corta en el percentil 99

            # Solo aplicamos zoom si el percentil 99 es mayor al mínimo (evita errores con constantes)
            if limite_superior > limite_inferior:
                plt.xlim(limite_inferior, limite_superior)

        plt.tight_layout()
        plt.show()