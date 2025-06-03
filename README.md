# 🧠 Projetos de Machine Learning - UFAL

Este repositório reúne pequenos projetos de aprendizado de máquina realizados para a disciplina de *Machine Learning*. Cada projeto explora diferentes técnicas de pré-processamento, classificação, agrupamento e avaliação de modelos, com conjuntos de dados diversos.

## 📁 Projetos

### 1. 🔬 Pré-processamento e Classificação de Diabetes

**Arquivo**: `diabetes_csv.py`

* Utiliza o dataset de diabetes para prever a presença da doença.
* Técnicas aplicadas:

  * Pré-processamento dos dados
  * Treinamento com K-Nearest Neighbors (KNN)
  * Envio das previsões para avaliação externa

### 2. 🐚 Classificação da Espécie de Abalone

**Arquivos**: `abalone_knn.py`, `abalone_florest.py`

* Classificação das espécies com dois modelos distintos:

  * KNN (`abalone_knn.py`)
  * Random Forest (`abalone_florest.py`)
* Aplicações adicionais:

  * OneHotEncoding para variáveis categóricas
  * Balanceamento de dados com SMOTE
  * Avaliação de acurácia
  * Envio para servidor externo de validação

### 3. 👁 Agrupamento por Características Oculares (KMeans)

**Arquivo**: `kmeans.py` (Barrett II)

* Agrupamento de perfis oculares com base em medidas biométricas.
* Aplicações:

  * Normalização com `StandardScaler`
  * Redução de dimensionalidade com PCA
  * Visualização com gráficos de dispersão
  * Método do cotovelo para definição de k ideal

### 4. 🧿 Identificação de Grupos por Espessura da Córnea (KMeans + Regras de Associação)

**Arquivo**: `kmeans.py` (RTVue)

* Clustering de pacientes a partir de espessuras corneanas.
* Técnicas utilizadas:

  * Tratamento de dados faltantes por média de gênero
  * Remoção de outliers com Z-Score
  * Agrupamento com KMeans (k=4)
  * Visualizações com PCA e t-SNE
  * Avaliação com Silhouette Score
  * Interpretação dos clusters com base em perfis médios

## ✅ Requisitos

* Python 3
* Bibliotecas: `pandas`, `numpy`, `sklearn`, `matplotlib`, `seaborn`, `imblearn`, `openpyxl`

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imblearn openpyxl
```

## 🛠 Como executar

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo
```

2. Execute o script desejado:

```bash
python nome_do_script.py
```

## 👨‍💻 Autor

Trabalhos realizados como parte da disciplina de Machine Learning - UFAL
**Aluno:** *João Victor Duarte do Nascimento*

