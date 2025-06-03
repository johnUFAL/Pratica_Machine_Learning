import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


path = '../RTVue_20221110.xlsx'
df = pd.read_excel(path)

media_genero = df.groupby('Gender')[['C', 'S', 'ST', 'T', 'IT',
                                     'I', 'IN', 'N', 'SN']].mean()

for colunas in ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']:
    df[colunas] = df.apply(
        lambda row: media_genero.loc[row['Gender'], colunas]
        if pd.isnull(row[colunas]) else row[colunas],
        axis = 1
    )

df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
df['Eye'] = df['Eye'].map({'OS': 0, 'OD': 1})

sns.pairplot(df[['Age', 'Gender', 'C', 'S', 'ST', 'T', 
                 'IT', 'I', 'IN', 'N', 'SN']], height=1.8)
#plt.show()
kind="hist"

var = ['Age', 'Gender', 'C', 'S', 'ST', 
       'T', 'IT', 'I', 'IN', 'N', 'SN']
df_select = df[var]
relacao = df_select.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(relacao, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlação")
plt.tight_layout()
#plt.show()

cols_pca = ['Age', 'Gender', 'Eye', 'C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']
x = df[cols_pca]

escala = StandardScaler()
x_normalizado = escala.fit_transform(x)

#PC
pca = PCA(n_components=4)
pca_result = pca.fit_transform(x_normalizado)

plt.figure(figsize=(8,6))
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1])
plt.title('PCA - Visualização de Clusters')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.show()

#Pearson
corr_pearson = df.corr(method='pearson')

#Spearman
corr_spearman = df.corr(method='spearman')

#lado a lado
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0])
axes[0].set_title('Correlação de Pearson')

sns.heatmap(corr_spearman, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1])
axes[1].set_title('Correlação de Spearman')

plt.tight_layout()
plt.show()