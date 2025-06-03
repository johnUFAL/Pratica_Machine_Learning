# PASSO 1: Configuração inicial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

#dados
path = 'RTVue_20221110.xlsx'
df = pd.read_excel(path)

#media por genero valor que falta
cols_espessura = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']
media_genero = df.groupby('Gender')[cols_espessura].mean()

for col in cols_espessura:
    df[col] = df.apply(
        lambda row: media_genero.loc[row['Gender'], col] if pd.isnull(row[col]) else row[col],
        axis=1
    )

#tratando outliers
z_scores = np.abs((df[cols_espessura] - df[cols_espessura].mean()) / df[cols_espessura].std())
df = df[(z_scores < 3).all(axis=1)].copy()

#normalizando
scaler = StandardScaler()
df[cols_espessura] = scaler.fit_transform(df[cols_espessura])

#k=4
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[cols_espessura])

#vizualizacao
def plot_clusters(features, title):
    plt.figure(figsize=(10, 6))
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = features[df['Cluster'] == cluster]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                   label=f'Cluster {cluster}', alpha=0.6)
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

#PCA
pca = PCA(n_components=2)
pca_features = pca.fit_transform(df[cols_espessura])
plot_clusters(pca_features, 'Clusters visualizados com PCA')

#TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_features = tsne.fit_transform(df[cols_espessura])
plot_clusters(tsne_features, 'Clusters visualizados com t-SNE')

#analise
print("\nTamanho de cada cluster:")
print(df['Cluster'].value_counts().sort_index())

print("\nCaracteristicas medias por cluster:")
cluster_means = df.groupby('Cluster')[cols_espessura].mean()
print(cluster_means)

#Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(cluster_means.T, cmap='YlGnBu', annot=True, fmt=".2f", linewidths=.5)
plt.title('Méedias Normalizadas das Características por Cluster')
plt.show()

#vendo qualidade 
silhouette = silhouette_score(df[cols_espessura], df['Cluster'])
print(f"\nSilhouette Score para k=4: {silhouette:.3f}")

#interpretaçao, pode ser melhor com mais k 
interpretation = {
    0: "Perfil com valores medios",
    1: "Perfil com valores abaixo da media",
    2: "Perfil com valores acima da media", 
    3: "Perfil com padrao misto/distinto"
}

print("\nInterpretação sugerida dos clusters:")
for cluster, desc in interpretation.items():
    print(f"Cluster {cluster}: {desc}")