import pandas as pd

def checar(df):
    print("Verificando..\n")

    nulos = df.isnull().sum()
    if nulos.any():
        print(nulos[nulos > 0])
    else:
        print("Nada\n")
    
    print("\n-----------------------\n")


path = 'barrettII_eyes_clustering.xlsx'
df = pd.read_excel(path)

checar(df)


#Nao encontrou nenhum valor NULL ou 0