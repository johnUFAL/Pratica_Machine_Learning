import pandas as pd
#codificação das var categoricas (sex)
from sklearn.preprocessing import OneHotEncodery
#nosso modelo
from sklearn.ensemble import RandomForestClassifier
#avaliação
from sklearn.metrics import accuracy_score
import requests

df = pd.read_excel("abalone_dataset.xlsx")
print(df.head())

#separar a var categorica 
encoder = OneHotEncodery(drop="first") #evita multicolinearidade 
sex_encoded = encoder.fit_transform(df[['Sex']]).toarray()

#DataFrame com novos valores 
sex_encoded_df = pd.DataFrame(sex_encoded, 
columns=encoder.get_feature_names_out(['Sex']))

#remove original e põe a nova 
df = df.drop(columns=['Sex']).reset_index(drop=True)
df = pd.concat([df, sex_encoded_df], axis=1)
print(df.head())

#separacao de atributos 
x = df.drop(columns=['Type']) #todas as colunas menos type
y = df['Type'] #classe alvo da previsao

#criacao do treino
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x, y)

#arquivo de teste
df_test = pd.read_excel("abalone_app.xlsx")

#aplicação das transformações
sex_encoded_test = encoder.transform(df_test[['Sex']].toarray())
sex_encoded_test_df = pd.DataFrame(sex_encoded_test,
columns=encoder.get_feature_names_out(['sex']))

df_test = df_test.drop(columns=['Sex']).reset_index(drop=True)
pd.concat([df_test, sex_encoded_test_df], axis=1)

#previsões
y_test_pred = model.predict(df_test)

#avaliação
accuracy = accuracy_score(y, model.predict(x))
print(f'Acurácia do modelo: {accuracy:.2f}')

#creditos prof
# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/03_validation.php"

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
