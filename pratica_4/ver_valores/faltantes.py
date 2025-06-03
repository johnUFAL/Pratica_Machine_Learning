import pandas as pd

def checar(df):
    print("Verificando..\n")

    nulos = df.isnull().sum()
    if nulos.any():
        print(nulos[nulos > 0])
    else:
        print("Nada\n")
    
    print("\n-----------------------\n")


path = '../RTVue_20221110.xlsx'
df = pd.read_excel(path)

checar(df)

##verificou-se alguns dados faltantes