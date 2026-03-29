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

#HISTOGRAMA con lista de variables
def histogramas(df, variables, target="is_fraud", bins=30):
    import matplotlib.pyplot as plt

    for col in variables:

        # Separar fraude vs legítimo
        fraude = df[df[target] == 1][col]
        legit = df[df[target] == 0][col]

        plt.figure(figsize=(16,9))

        # Histogramas
        plt.hist(legit, bins=bins, alpha=0.6, label="Legítimo", edgecolor='black')
        plt.hist(fraude, bins=bins, alpha=0.6, label="Fraude", edgecolor='black')

        # Etiquetas
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
        plt.title(f"{col}: Fraude vs Legítimo")
        plt.legend()

        # Grilla
        plt.grid(axis='y', visible=True)
        plt.grid(axis='x', visible=False)

        plt.tight_layout()
        plt.show()