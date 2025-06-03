import pandas as pd

df = pd.read_excel('barrettII_eyes_clustering.xlsx')

#miopia
con_mio_AL = df['AL'] > 18
con_mio_K2 = df['K2'] > 42

total = len(df)

sup_mio = len(df[con_mio_AL & con_mio_K2]) / total
print(f'Suporte (Miopia): {sup_mio:.2%}')

if len(df[con_mio_AL]) > 0:
    conf_mio = len(df[con_mio_AL & con_mio_K2]) / len(df[con_mio_AL])
    print(f'Confiança (Miopia): {conf_mio:.2%}')

sup_A = len(df[con_mio_AL]) / total
sup_B = len(df[con_mio_K2]) / total
if sup_A * sup_B > 0:
    lift_mio = sup_mio / (sup_A * sup_B)
    print(f'LIFT (Miopia): {lift_mio:.2f}')

#hipermetropia
con_hiper_AL = df['AL'] < 18
con_hiper_K1 = df['K1'] < 42

sup_hiper = len(df[con_hiper_AL & con_hiper_K1]) / total
print(f'Suporte (Hipermetropia): {sup_hiper:.2%}')

if len(df[con_hiper_AL]) > 0:
    conf_hiper = len(df[con_hiper_AL & con_hiper_K1]) / len(df[con_hiper_AL])
    print(f'Confiança (Hipermetropia): {conf_hiper:.2%}')

sup_A = len(df[con_hiper_AL]) / total
sup_B = len(df[con_hiper_K1]) / total
if sup_A * sup_B > 0:
    lift_hiper = sup_hiper / (sup_A * sup_B)
    print(f'LIFT (Hipermetropia): {lift_hiper:.2f}')
