import pandas as pd
import numpy as np
#codificação das var categoricas (sex)
from sklearn.preprocessing import OneHotEncoder
#nosso modelo
from sklearn.ensemble import RandomForestClassifier
#avaliação
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
import requests
from imblearn.over_sampling import SMOTE

df = pd.read_excel("abalone_dataset.xlsx")
print(df.head())

#separar a var categorica 
encoder = OneHotEncoder(drop="first") #evita multicolinearidade
sex_encoded = encoder.fit_transform(df[['sex']]).toarray()

#DataFrame com novos valores 
sex_encoded_df = pd.DataFrame(sex_encoded, 
columns=encoder.get_feature_names_out(['sex']))

#remove original e põe a nova 
df = df.drop(columns=['sex']).reset_index(drop=True)
df = pd.concat([df, sex_encoded_df], axis=1)
print(df.head())

#separacao de atributos 
x = df.drop(columns=['type']) #todas as colunas menos type
y = df['type'] #classe alvo da previsao

#dividindo teste de treino
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#para equilibara as classes no treino
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

#criacao do treino
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(x_train, y_train)

#balanceamento para nao favorecer classes mais recorrentesS
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo no conjunto de teste: {accuracy:.2f}')

#arquivo de teste
df_test = pd.read_excel("abalone_app.xlsx")

#aplicação das transformações
sex_encoded_test = encoder.transform(df_test[['sex']]).toarray()
sex_encoded_test_df = pd.DataFrame(sex_encoded_test,
columns=encoder.get_feature_names_out(['sex']))

df_test = df_test.drop(columns=['sex']).reset_index(drop=True)
df_test = pd.concat([df_test, sex_encoded_test_df], axis=1)

#previsões
y_test_pred = model.predict(df_test)

#avaliação
accuracy = accuracy_score(y, model.predict(x))
print(f'Acurácia do modelo: {accuracy:.2f}')

#creditos prof
# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/03_Validation.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "Deltinha"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions': pd.Series(y_test_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")
