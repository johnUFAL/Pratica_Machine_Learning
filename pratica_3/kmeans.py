import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#dados 
path = 'barrettII_eyes_clustering.xlsx'
df = pd.read_excel(path)

#media entre k1 e k2
df['k_avg'] = (df['K1'] + df['K2']) / 2

#pegar colunas dps da media
var = ['AL', 'ACD', 'WTW', 'k_avg']
dados = df[var]

#normalizando os dados 
escala = StandardScaler()
dados_normalizados = escala.fit_transform(dados)

#metodo cotovelo para k
inercia = []
k_raio = range(1, 11)

for k in k_raio:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(dados_normalizados)
    inercia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_raio, inercia, marker='o')
plt.xlabel('Num Clusters (k)')
plt.ylabel('Inercia')
plt.title('Cotovelo')
plt.grid(True)
#plt.show()

#hora de escolher um k bom e pelo gráfico é 2
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(dados_normalizados)

#rotulos ao original
df['Cluster'] = clusters

#aplicando PCA
pca = PCA(n_components=2) 
dados_pca = pca.fit_transform(dados_normalizados)

#cluster com PCA
plt.figure(figsize=(8, 6))
plt.scatter(dados_pca[:,0], dados_pca[:,1], c=clusters, cmap='rainbow', s=50)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clusters com PCA')
plt.grid(True)
plt.show()

#cluster com Scatter Plot 2D: Al vs k_avg
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='AL', y='k_avg', hue='Cluster', palette='rainbow', s=70)
plt.title('AL vs k_avg')
plt.grid(True)
plt.show()

#cluster com Scatter Plot 2D:  WTW vs ACD
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='WTW', y='ACD', hue='Cluster', palette='rainbow', s=70)
plt.title('WTW vs ACD')
plt.grid(True)
plt.show()
