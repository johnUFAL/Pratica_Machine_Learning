import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import requests
import numpy as np

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_excel('abalone_dataset_corrigido.xlsx')

# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['sex','length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight']
X = data[feature_cols]
y = data.type

# Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_excel('abalone_app_corrigido.xlsx')
data_app = data_app[feature_cols]

y_pred = neigh.predict(data_app)

y_pred = neigh.predict(data_app)

y_train_pred = neigh.predict(X)
print("Acurácia na base de treino:", accuracy_score(y, y_train_pred))

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/03_Validation.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "Deltinha"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")