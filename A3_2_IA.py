import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Dataset
df = pd.read_csv(
    "C:\\Users\\minec\\OneDrive\\Área de Trabalho\\IA\\smartphone_dataset_pt_br.csv")

# Remove colunas que não são necessárias para o cálculo
df = df.drop(columns=['Resolução'])

# Preenche valores ausentes para não interferir
df = df.dropna()

# Cópia para preservar os dados originais
df_original = df.copy()

# Normaliza as colunas de interesse para cálculo de custo-benefício
features_to_normalize = ['Avaliação', 'Preço', 'Veloc_Processador',
                         'Capac_Bateria', 'Memória_Interna', 'Capacidade_Ram']
scaler = MinMaxScaler()
df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

# Pesos
weights = {
    'Avaliação': 0.3,        # Avaliação tem peso alto
    'Preço': -0.2,           # Preço é negativo, pois menor preço é melhor
    'Veloc_Processador': 0.2,
    'Capac_Bateria': 0.1,
    'Memória_Interna': 0.1,
    'Capacidade_Ram': 0.1
}

# Cálculo do índice de custo-benefício
df_original['Custo_Benefício'] = (
    df['Avaliação'] * weights['Avaliação'] +
    df['Preço'] * weights['Preço'] +
    df['Veloc_Processador'] * weights['Veloc_Processador'] +
    df['Capac_Bateria'] * weights['Capac_Bateria'] +
    df['Memória_Interna'] * weights['Memória_Interna'] +
    df['Capacidade_Ram'] * weights['Capacidade_Ram']
)

# Exibe os 10 melhores celulares em custo-benefício com dados originais
melhores_celulares = df_original.sort_values(
    by='Custo_Benefício', ascending=False)
print("Top 10 celulares em custo-benefício:")
print(melhores_celulares[['Marca', 'Modelo',
      'Avaliação', 'Preço', 'Veloc_Processador', 'Capac_Bateria', 'Memória_Interna', 'Capacidade_Ram', 'Custo_Benefício']].head(10))
