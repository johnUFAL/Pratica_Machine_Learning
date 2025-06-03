import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#dados
path = 'abalone_dataset.xlsx'
df = pd.read_excel(path)

#histograma das var de num
df[['length', 'diameter', 'height', 'whole_weight', 
    'shucked_weight', 'viscera_weight', 'shell_weight']].hist(figsize=(12, 8), bins=20)
plt.show()

#dispersão entre comprimento e peso total
sns.scatterplot(x=df['length'], y=df['whole_weight'], hue=df['sex'])
plt.title("Comprimento vs. Peso Total")
plt.xlabel("Length (mm)")
plt.ylabel("Whole weight (g)")
plt.show()

#correlação entre todas as var num
df_corr = df.select_dtypes(include=['number'])

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriz de Correlação")
plt.show()

#comparação do comprimento médio por sexo
df.groupby('sex')['length'].mean().plot(kind='bar', color=['blue', 'red', 'green'])
plt.title('Média do Comprimento por Sexo')
plt.xlabel('Sexo')
plt.ylabel('Comprimento Médio (mm)')
plt.show()

#distribuição do peso total por tipo de abalone
plt.figure(figsize=(10, 5))
sns.violinplot(x=df['type'], y=df['whole_weight'])
plt.title('Distribuição do Peso Total por Tipo de Abalone')
plt.xlabel('Tipo')
plt.ylabel('Peso Total (g)')
plt.show()
