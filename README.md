<h1 align="center">Trabalho Prático IA (2025/2)</h1>

<div align="center">

![Python](https://img.shields.io/badge/python-blue?style=for-the-badge&logo=python&logoColor=white)
![VS Code](https://img.shields.io/badge/visual%20studio%20code-blue?style=for-the-badge)


📖:
 [Visão Geral](#visão-geral) |
[Como reproduzir](#como-reproduzir) |
[Decisões Técnicas](#decisões-técnicas)

</div>


## Visão Geral

<div align="justify">
O objetivo do trabalho, realizado na disciplina de Inteligência Artificial ofertada pelo professor Tiago Aves de Oliveira, foi de compreender, implementar e comparar algoritmos clássicos de IA e Computação Natural, preparando e analisando dados reais.
</div>

### Partes do trabalho:
O trabalho foi dividido em quatro partes:

- Parte 1 - Árvore de decisão manual
- Parte 2 - Supervisionado (Kaggle/UCI): KNN, SVM e Árvore 
- Parte 3 - Algoritmo Genético (AG)
- Parte 4 - Enxame e Imunes

### Estrutura do Repositório:
```
TRABALHO PRÁTICO IA/
│   .gitattributes
│   README.md
│   requirements.txt
│   svm.model
│   
├───data
│   ├───processed
│   │       benchmark_results.csv
│   │       comparison_report.txt
│   │       confusion_matrix_dt_100000.png
│   │       confusion_matrix_knn_100000.png
│   │       confusion_matrix_svm_100000.png
│   │       decision_tree_visualization.png
│   │       decision_tree_visualization_100000.png
│   │       X_test.csv
│   │       X_test_scaled.csv
│   │       X_train.csv
│   │       X_train_scaled.csv
│   │       y_test.csv
│   │       y_train.csv
│   │
│   └───raw
│           Watera.csv
│
└───src
    ├───part1_tree_manual
    │       tree_diagram.md
    │       tree_image.png
    │       tree_manual.py
    │
    ├───part2_ml
    │   │   preprocess.py
    │   │   train_knn.py
    │   │   train_svm.py
    │   │   train_tree.py
    │   │   util_metrics.py
    │   │
    │   └───__pycache__
    │           preprocess.cpython-310.pyc
    │           preprocess.cpython-311.pyc
    │           util_metrics.cpython-310.pyc
    │
    └───part3_ga
            ga.py
```

<!--
```
TREE-DECISION/
│
├── data/
│   ├── raw/                    # Dados brutos
│   │   └── plant_growth_data.csv
│   └── processed/              # Dados processados
│       ├── X_train.csv         # Features de treino (sem escalonamento)
│       ├── X_train_scaled.csv  # Features de treino (escalonadas)
│       ├── y_train.csv         # Target de treino
│       └── ...
│
├── src/
│   ├── part1_tree_manual/      # Implementação manual
│   │   └── tree_manual.py      # Classe Tree customizada [Exemplo: 32 correntes filosóficas]
│   │   
│   │
│   └── part2_ml/               # Machine Learning
│       ├── preprocess.py       # Pré-processamento completo
│       ├── train_tree.py       # Treinar Decision Tree
│       ├── train_knn.py        # Treinar KNN
│       ├── train_svm.py        # Treinar SVM
│       └── util_metrics.py     # Funções de métricas
│
├── requirements.txt            # Dependências
└── README.md                   # Este arquivo
```
-->

<!--
Este projeto explora **Árvores de Decisão** de duas formas:

1. **Parte 1 - Implementação Manual (`src/part1_tree_manual/`)**
   - Estrutura de dados de árvore binária do zero
   - Visualização com NetworkX e Matplotlib
   - Exemplo: Árvore de decisão filosófica com 32 correntes (6 níveis)

2. **Parte 2 - Machine Learning (`src/part2_ml/`)**
   - Pré-processamento robusto de dados
   - Treinamento de modelos: Decision Tree, KNN, SVM
   - Métricas de avaliação e validação cruzada 
-->



## Como reproduzir
### Pré-requisitos

- **Python 3.8+** instalado
- **pip** (gerenciador de pacotes Python)
- **Git** (opcional, para clonar o repositório)

### Instalar Dependências

```bash
pip install -r requirements.txt
```

---

### Bibliotecas Utilizadas

| Biblioteca | Versão | Por Que Usamos? |
|------------|--------|-----------------|
| **pandas** | 2.1.4 | Manipulação de dados tabulares (CSV, DataFrames) |
| **numpy** | 1.26.3 | Operações numéricas eficientes (arrays, matrizes) |
| **scikit-learn** | 1.3.2 | **Biblioteca principal de ML**: Decision Tree, KNN, SVM, pré-processamento, métricas |
| **scipy** | 1.11.4 | Algoritmos científicos (dependência do scikit-learn) |

<!--
### Sobre a escolha de cada Biblioteca

#### **scikit-learn** 
```python
# Modelos de ML
from sklearn.tree import DecisionTreeClassifier       # Árvore de Decisão
from sklearn.neighbors import KNeighborsClassifier    # KNN
from sklearn.svm import SVC                           # SVM

# Pré-processamento
from sklearn.preprocessing import StandardScaler      # Escalonamento (KNN/SVM)
from sklearn.preprocessing import LabelEncoder        # String → Número

# Métricas e Validação
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
```

#### **pandas** + **numpy**
```python
import pandas as pd    # Ler CSV, manipular tabelas
import numpy as np     # Operações matemáticas rápidas
```
-->

## Decisões Técnicas

### **Parte 1: Árvore Manual (Filosófica)**

```bash
cd src/part1_tree_manual
python3 tree_manual.py
```

**O que faz:**
- Sistema interativo com 6 níveis de perguntas
- Identifica 32 correntes filosóficas
- Recomendações de livros personalizadas
- Visualização colorida por área filosófica

---

### **Parte 2: Machine Learning**

#### **1. Pré-processar Dados**

```bash
cd src/part2_ml
python3 preprocess.py
```

**O que faz:**
- Trata valores nulos (mediana para numéricos, moda para categóricos)
- Label Encoding (string → número)
- Escalonamento (StandardScaler para KNN/SVM)
- Divisão estratificada treino/teste (80/20)
- Salva dados processados em `data/processed/`

#### **2. Treinar Modelos**

```bash
# Árvore de Decisão
python3 train_tree.py

# KNN (K-Nearest Neighbors)
python3 train_knn.py

# SVM (Support Vector Machine)
python3 train_svm.py
```

---

#### Pré-processamento Detalhado

##### **O que o `preprocess.py` faz?**

1. **Valores Nulos:**
   - Numéricos: preenche com **mediana** (robusto a outliers)
   - Categóricos: preenche com **moda** (valor mais frequente)
2. **Label Encoding:**
   - Transforma strings em números: `['loam', 'sandy', 'clay']` → `[0, 1, 2]`
   - **Necessário** porque ML trabalha apenas com números
3. **One-Hot Encoding (opcional):**
   - Cria colunas binárias para categorias sem ordem natural
   - Exemplo: `Soil_Type='loam'` → `Soil_Type_loam=1, Soil_Type_sandy=0`
4. **Escalonamento (StandardScaler):**
   - Padroniza: **média=0, desvio=1**
   - **Essencial** para KNN e SVM (sensíveis à magnitude)
   - **NÃO necessário** para Árvores de Decisão
5. **Divisão Estratificada:**
   - Mantém proporção das classes em treino/teste
   - Se 70% é classe A, treino terá 70% classe A
6. **Validação Cruzada (K-Fold):**
   - Divide dados em K partes (5 por padrão)
   - Treina K vezes para estimativa mais confiável

---

