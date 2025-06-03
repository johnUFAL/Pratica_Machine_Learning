import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats

#dados do dataframe
path = 'RTVue_20221110.xlsx'
df = pd.read_excel(path)

#media por gênero para dados faltantes 
media_genero = df.groupby('Gender')[['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']].mean()

for coluna in ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']:
    df[coluna] = df.apply(
        lambda row: media_genero.loc[row['Gender'], coluna]
        if pd.isnull(row[coluna]) else row[coluna],
        axis=1
    )

#tratando dados categoricos
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
encoder = OneHotEncoder(sparse_output=False, drop='first')
eye_encoded = encoder.fit_transform(df[['Eye']])
df['Eye_OD'] = eye_encoded[:, 0]  # OD=1, OS=0
df.drop('Eye', axis=1, inplace=True)

#selecionando colunas
cols_espessura = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']
x = df[cols_espessura].copy()

#analise de inicio
print("\nEstatisticas ANTES do tratamento:")
print(x.describe())

plt.figure(figsize=(12,6))
x.boxplot()
plt.xticks(rotation=45)
plt.title('Boxplot ANTES do tratamento')
plt.show()

#procurando outliers
print("\nProcurando outliers...")
z_scores = np.abs(stats.zscore(x))
outliers = (z_scores > 3).any(axis=1)
print(f"Num de outliers: {outliers.sum()}")

#grafico para ver outliers
plt.figure(figsize=(10,6))
sns.scatterplot(x=x.index, y=x['S'], hue=outliers, palette={True: 'red', False: 'blue'})
plt.title('Identificacao de outliers (exemplo: S)')
plt.show()

#remoçao deles
x_clean = x[~outliers]
df_clean = df[~outliers].copy()

print("\nEstatisticas APOS remover outliers:")
print(x_clean.describe())

#normalizaçao
escala = StandardScaler()
dados_normalizados = escala.fit_transform(x_clean)
df_clean[cols_espessura] = dados_normalizados

#verifica apos 
print("\nVerificacao da normalizacao:")
print(pd.DataFrame(dados_normalizados, columns=cols_espessura).describe())

#ver distribuicao apos normalizar
plt.figure(figsize=(12,6))
pd.DataFrame(dados_normalizados, columns=cols_espessura).boxplot()
plt.xticks(rotation=45)
plt.title('Distribuicao das colunas apos normalizacao')
plt.show()

#kmeans
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(dados_normalizados)
df_clean['Cluster'] = clusters

#PCA
pca = PCA(n_components=2)
dados_pca = pca.fit_transform(dados_normalizados)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(dados_pca[:,0], dados_pca[:,1], c=clusters, cmap='viridis', alpha=0.6)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Clusters no espaço PCA')

#TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
dados_tsne = tsne.fit_transform(dados_normalizados)

plt.subplot(1,2,2)
plt.scatter(dados_tsne[:,0], dados_tsne[:,1], c=clusters, cmap='viridis', alpha=0.6)
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('Clusters no espaço t-SNE')

plt.tight_layout()
plt.show()

#analise
print("\nMedias por cluster:")
cluster_means = df_clean.groupby('Cluster')[cols_espessura].mean()
print(cluster_means)

#vendo media
plt.figure(figsize=(12,6))
sns.heatmap(cluster_means.T, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title('Medias normalizadas por cluster')
plt.show()