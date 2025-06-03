import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

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

#hora de escolher um k bom e pelo gráfico é 2
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(dados_normalizados)
df['rótulo_kmeans'] = clusters

# --------------------------------------
#regras 
df['diagnostico'] = 'normal'
df.loc[(df['AL'] > 18) & (df['K2'] > 42), 'diagnostico'] = 'miopia'
df.loc[(df['AL'] < 18) & (df['K1'] < 42), 'diagnostico'] = 'hipermetropia'

cross_tab = pd.crosstab(df['rótulo_kmeans'], df['diagnostico'])
print(cross_tab)

sns.countplot(x='rótulo_kmeans', hue='diagnostico', data=df)
plt.show()
