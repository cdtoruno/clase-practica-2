import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

url = 'https://raw.githubusercontent.com/cdtoruno/new-repo/main/Spotify_Youtube.csv'
data = pd.read_csv(url)

# Exploración inicial del dataset
print(data.head())
print(data.shape)
print(data.info())
print(data.describe())

df_original = data.copy()

# Identificar valores faltantes
print('\nValores faltantes por columna:')
missing_values = data.isnull().sum()
print(missing_values[missing_values > 0])

# Seleccionar las columnas importantes
columnas_importantes = ['Energy', 'Views', 'Likes', 'Comments', 'Stream', 'Licensed', 'official_video']

# Crear un nuevo DataFrame con estas columnas seleccionadas
datos_importantes = data[columnas_importantes].copy()

# Imputar valores faltantes con la mediana para las columnas numéricas
for column in ['Energy', 'Views', 'Likes', 'Comments', 'Stream']:
    datos_importantes.loc[:, column] = datos_importantes.loc[:, column].fillna(datos_importantes[column].median())

# Imputar valores faltantes con la moda para las columnas categóricas
for column in ['Licensed', 'official_video']:
    datos_importantes.loc[:, column] = datos_importantes.loc[:, column].fillna(datos_importantes[column].mode()[0])

# Boxplot antes de eliminar los outliers
def plot_boxplots_before_after(data_before, data_after, columns):
    for column in columns:
        plt.figure(figsize=(14, 6))

        # Boxplot antes de eliminar outliers
        plt.subplot(1, 2, 1)
        sns.boxplot(x=data_before[column])
        plt.title(f'{column}: Boxplot con outliers')

        # Boxplot después de eliminar outliers
        plt.subplot(1, 2, 2)
        sns.boxplot(x=data_after[column])
        plt.title(f'{column}: Boxplot sin outliers')
        plt.show()

# Aplicar Z-score para detección de outliers en las columnas numéricas
z_scores = np.abs(stats.zscore(datos_importantes[['Energy', 'Views', 'Likes', 'Comments', 'Stream']]))
threshold = 3
outliers_condition = (z_scores < threshold).all(axis=1)

# Crear un DataFrame sin outliers
datos_sin_outliers = datos_importantes[outliers_condition]

# Mostrar boxplots antes y después de eliminar outliers
plot_boxplots_before_after(datos_importantes, datos_sin_outliers, ['Energy', 'Views', 'Likes', 'Comments', 'Stream'])

# Normalización (Min-Max Scaling) para las columnas numéricas
scaler_minmax = MinMaxScaler()
minmax_scaled_data = scaler_minmax.fit_transform(datos_sin_outliers[['Energy', 'Views', 'Likes', 'Comments', 'Stream']])
minmax_scaled_df = pd.DataFrame(minmax_scaled_data, columns=['Energy', 'Views', 'Likes', 'Comments', 'Stream'])

# Estandarización (Z-score Scaling) para las columnas numéricas
scaler_standard = StandardScaler()
zscore_scaled_data = scaler_standard.fit_transform(datos_sin_outliers[['Energy', 'Views', 'Likes', 'Comments', 'Stream']])
zscore_scaled_df = pd.DataFrame(zscore_scaled_data, columns=['Energy', 'Views', 'Likes', 'Comments', 'Stream'])

# Imprimir los primeros resultados de Min-Max Scaling y Z-score Scaling
print('\n Min-Max \n')
print(minmax_scaled_df.head())

print('\nZ-Score\n')
print(zscore_scaled_df.head())

# Creación de nuevas variables
# Proporción de Likes y Comments sobre Views
datos_sin_outliers['Engagement_Rate'] = (datos_sin_outliers['Likes'] + datos_sin_outliers['Comments']) / datos_sin_outliers['Views']

# Proporción de Likes y Comments sobre Stream
datos_sin_outliers['Interaction_Score'] = (datos_sin_outliers['Likes'] + datos_sin_outliers['Comments']) / datos_sin_outliers['Stream']

print(datos_sin_outliers[['Engagement_Rate']].head())
print(datos_sin_outliers[['Interaction_Score']].head())

# Codificación de variables categóricas (One-Hot Encoding)
print('\nCodificación de variables categóricas')
datos_codificados = pd.get_dummies(datos_sin_outliers, columns=['Licensed', 'official_video'], drop_first=True)
print(datos_codificados.head())

# Comparación de las estadísticas antes y después del preprocesamiento.
print("\nAntes del preprocesamiento:")
print(df_original.describe())

print("\nDespués del preprocesamiento:")
print(datos_codificados.describe())
