import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#dados 'AL,	ACD,	WTW,	K1, 	K2'
path = 'barrettII_eyes_clustering.xlsx'
df = pd.read_excel(path)

#sns.pairplot(df[['AL',	'ACD',	'WTW',	'K1', 	'K2']], height=1.8)
#plt.show()
#kind="hist"

var = ['AL', 'ACD',	'WTW', 'K1', 'K2']
df_select = df[var]
relacao = df_select.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(relacao, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlação")
plt.tight_layout()
plt.show()
