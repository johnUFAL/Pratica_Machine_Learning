import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

path = '../RTVue_20221110.xlsx'
df = pd.read_excel(path)

cols_espessura = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']
media_genero = df.groupby('Gender')[cols_espessura].mean()

for col in cols_espessura:
    df[col] = df.apply(
        lambda row: media_genero.loc[row['Gender'], col] if pd.isnull(row[col]) else row[col],
        axis=1
    )

z_scores = np.abs((df[cols_espessura] - df[cols_espessura].mean()) / df[cols_espessura].std())
df = df[(z_scores < 3).all(axis=1)].copy()

scaler = StandardScaler()
df[cols_espessura] = scaler.fit_transform(df[cols_espessura])

kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[cols_espessura])

#os angulos das regioes corneanas
angles = {
    'C': (0, 0),        
    'S': (0, 1),        
    'ST': (0.707, 0.707),  
    'T': (1, 0),       
    'IT': (0.707, -0.707),
    'I': (0, -1),      
    'IN': (-0.707, -0.707), 
    'N': (-1, 0),       
    'SN': (-0.707, 0.707),
}

#media valor att
cluster_means = df.groupby('Cluster')[cols_espessura].mean()

#obtendo valor reais
mean_values = df[cols_espessura].mean()  #media real
std_values = df[cols_espessura].std()    #dp REAL
cluster_means_real = cluster_means * std_values + mean_values

def create_topographic_map(cluster_data, cluster_num):
    #preenchendo dados
    x = [angles[region][0] for region in cols_espessura]
    y = [angles[region][1] for region in cols_espessura]
    z = cluster_data[cols_espessura].values
    
    #fazendo grid
    xi = yi = np.linspace(-1.5, 1.5, 200)
    xi, yi = np.meshgrid(xi, yi)
    
    #interpolacao usando cubic
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    
    #mapa de calor 2D
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(xi, yi, zi, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Espessura (normalizada)')
    
    #marcar regioes originais 
    plt.scatter(x, y, c='red', s=50, edgecolors='black')
    for i, region in enumerate(cols_espessura):
        plt.text(x[i], y[i], region, ha='center', va='center', fontweight='bold')
    
    plt.title(f'Mapa Topográfico - Cluster {cluster_num}\n{interpretation[cluster_num]}')
    plt.axis('equal')
    plt.show()
    
    #em 3D
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', 
                         rstride=1, cstride=1, alpha=0.8, 
                         antialiased=True)
    fig.colorbar(surf, label='Espessura (normalizada)')
    ax.set_title(f'Topografia 3D - Cluster {cluster_num}\n{interpretation[cluster_num]}')
    ax.set_xlabel('Posição X')
    ax.set_ylabel('Posição Y')
    plt.show()

#interpretacao, pode ser melhor com mais k 
interpretation = {
    0: "Perfil com valores medios",
    1: "Perfil com valores abaixo da media",
    2: "Perfil com valores acima da media", 
    3: "Perfil com padrao misto/distinto"
}

#media/cluters
cluster_means = df.groupby('Cluster')[cols_espessura].mean()

#mapa para cada k
for cluster_num in cluster_means.index:
    create_topographic_map(cluster_means.loc[cluster_num], cluster_num)