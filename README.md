<h1 align="center">Trabalho Prático IA [Algoritmos] (2025/2)</h1>

<div align="center">

![Python](https://img.shields.io/badge/python-blue?style=for-the-badge&logo=python&logoColor=white)
![VS Code](https://img.shields.io/badge/visual%20studio%20code-blue?style=for-the-badge)
![Ubuntu](https://img.shields.io/badge/ubuntu-orange?style=for-the-badge&logo=ubuntu&logoColor=white)

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

<!--
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
-->
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

---

### Instalação Rápida

#### Usando run.sh (Linux):
```bash
# Concede permissões de execução:
chmod +x run.sh

# Cria .venv e baixa depenências nele:
./run.sh
```

#### Usando Makefile (Alternativa):

```bash
# Instalar dependências
make install

# Executar Parte 1 (Árvore Manual Filosófica)
make part1

# Executar Parte 2 completa (ML: pré-processamento + treinos)
make part2

# Ou executar partes específicas:
make part2-dt         # Apenas Decision Tree
make part2-knn        # Apenas KNN
make part2-svm        # Apenas SVM

# Executar Parte 3 (Algoritmo Genético)
make part3

# Executar Parte 4 (Enxame e Imunes)
make part4              # Executa ACO e CLONALG
make part4-aco          # Apenas ACO vs GA
make part4-clonalg      # Apenas CLONALG vs ACO vs GA

# Ver resultados
make results

# Limpar arquivos gerados
make clean
```

#### Instalação Manual

```bash
pip install -r requirements.txt
```

---

### Bibliotecas Utilizadas

| Biblioteca | Versão | Por Que Usamos? | Decisão de Implementação |
|------------|--------|-----------------|--------------------------|
| **pandas** | 2.1.4 | Manipulação de dados tabulares (CSV, DataFrames) | Escolhido por sua eficiência em operações de leitura/escrita de CSV e transformações de dados. A API intuitiva permite operações complexas em uma linha. |
| **numpy** | 1.26.3 | Operações numéricas eficientes (arrays, matrizes) | Base para computação científica em Python. Vetorização de operações acelera cálculos em 10-100x comparado a loops Python puros. |
| **scikit-learn** | 1.3.2 | **Biblioteca principal de ML**: Decision Tree, KNN, SVM, pré-processamento, métricas | API consistente entre algoritmos facilita experimentação. Implementações otimizadas em C/Cython garantem performance. Amplamente testada e documentada. |
| **scipy** | 1.11.4 | Algoritmos científicos e estatísticos (dependência do scikit-learn) | Fornece estruturas de dados especializadas (sparse matrices) e algoritmos de otimização usados internamente pelo scikit-learn. |
| **matplotlib** | 3.8.2 | Visualização de dados (gráficos, plots) | Padrão *de facto* para visualização científica em Python. Flexibilidade para customização detalhada de gráficos. |
| **seaborn** | 0.13.0 | Visualização estatística (matriz de confusão) | Built on top do matplotlib, oferece temas visuais mais modernos e funções especializadas para análise estatística. |

---

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

#### Execução

```bash
make part1
```

#### O que faz
- Sistema interativo com 6 níveis de perguntas filosóficas
- Identifica 32 correntes filosóficas diferentes
- Recomendações de livros personalizadas (200+ obras catalogadas)

#### Decisões de Implementação

**1. Estrutura de Dados Customizada**
```python
class Tree:
    def __init__(self, question, left=None, right=None, leaf_value=None):
        self.question = question
        self.left = left
        self.right = right
        self.leaf_value = leaf_value
```
- **Por que classe própria?** Controle total sobre a lógica de navegação e visualização. Permite adicionar métodos customizados (ex: `collect_all_nodes()`, `get_depth()`).
- **Alternativa considerada:** Usar `sklearn.tree.DecisionTreeClassifier`, mas não permite construção manual da árvore com perguntas textuais.

**2. Hierarquia de 6 Níveis**
```
Nível 1: Conhecimento (Racionalismo vs Empirismo)
Nível 2: Realidade (Materialismo vs Idealismo)
Nível 3: Ética (Deontologia vs Consequencialismo)
Nível 4: Existência (Determinismo vs Livre-arbítrio)
Nível 5: Política (Individualismo vs Coletivismo)
Nível 6: Estética (Objetividade vs Subjetividade)
```
- **Por que 6 níveis?** 2^6 = 64 folhas possíveis, mas usamos 32 correntes (algumas folhas compartilham ramos). Profundidade equilibra especificidade com usabilidade.
- **Decisão de design:** Ordem das perguntas segue progressão lógica (fundamentos → aplicações práticas).


---

### **Parte 2: Machine Learning (Supervisionado)**

#### Execução Completa

```bash
# Execução de todas os algoritmos:
make part2

# Execução individual dos algoritmos:
make part2-preprocess
make part2-dt
make part2-knn
make part2-svm
```

---

#### **1. Pré-processamento (`preprocess.py`)**

##### Decisões de Implementação

**A. Tratamento de Valores Nulos**
```python
# Numéricos: Mediana (não média!)
imputer_num = SimpleImputer(strategy='median')
X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])

# Categóricos: Moda
imputer_cat = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])
```
- **Por que mediana e não média?** 
  - Média é sensível a outliers. Se 99 valores são ~10 e 1 valor é 10.000, a média será distorcida.
  - Mediana é robusta: sempre retorna o valor central.
  - **Exemplo real:** Em dados de qualidade de água, um sensor defeituoso pode gerar pH=999. Mediana ignora esse erro.

**B. Label Encoding vs One-Hot Encoding**
```python
# Label Encoding (usado no projeto)
le = LabelEncoder()
X['Soil_Type'] = le.fit_transform(X['Soil_Type'])
# ['loam', 'sandy', 'clay'] → [0, 1, 2]

# One-Hot Encoding (comentado, mas disponível)
X_encoded = pd.get_dummies(X, columns=['Soil_Type'], drop_first=True)
# Soil_Type_loam | Soil_Type_sandy | Soil_Type_clay
#       1        |        0        |       0
```
- **Por que Label Encoding?**
  - **Decision Trees:** Não precisam de One-Hot. Árvores podem lidar com ordinais (0, 1, 2) diretamente.
  - **KNN/SVM:** Label Encoding funciona quando há ordem natural (ex: Pequeno=0, Médio=1, Grande=2).
  - **Quando usar One-Hot:** Categorias sem ordem (ex: cores: vermelho, azul, verde). Evita o modelo assumir que "azul" (1) está "entre" vermelho (0) e verde (2).
- **Trade-off:** One-Hot aumenta dimensionalidade. 10 categorias → 10 colunas. Afeta performance em datasets grandes.

**C. Escalonamento: StandardScaler vs MinMaxScaler**
```python
# StandardScaler (média=0, desvio=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler (valores entre 0 e 1) - usado no projeto
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```
- **Por que MinMaxScaler?**
  - **KNN:** Distâncias euclidianas são afetadas por escala. Feature "Renda" (0-100k) dominaria "Idade" (0-100).
  - **SVM:** Kernel RBF usa distâncias. Features com ranges diferentes quebram a simetria do kernel.
  - **MinMaxScaler vs StandardScaler:** 
    - MinMaxScaler preserva distribuição original (boa para dados sem outliers extremos).
    - StandardScaler melhor quando há outliers (normaliza pelo desvio padrão).
  - **Decision Trees NÃO precisam:** Árvores fazem splits baseados em thresholds relativos (ex: "pH > 7?"). Escala não importa.

**D. Divisão Estratificada**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,  # CRUCIAL!
    random_state=42
)
```
- **Por que estratificação?**
  - **Problema:** Dataset com 90% classe A, 10% classe B. Split aleatório pode gerar treino com 95% A, teste com 85% A.
  - **Solução:** `stratify=y` garante que treino E teste tenham 90% A, 10% B.
  - **Impacto:** Sem estratificação, métricas podem ser enviesadas. Modelo "aprende" distribuição diferente da realidade.

**E. Amostragem Aleatória (10k, 50k, 100k)**
```python
df_sample = df.sample(n=sample_size, random_state=42)
```
- **Por que amostrar?**
  - **Benchmarking:** Comparar performance dos algoritmos em diferentes escalas de dados.
  - **Trade-off tempo vs acurácia:** SVM com 100k linhas pode levar horas. 10k linhas permite iteração rápida.
  - **`random_state=42`:** Reprodutibilidade. Mesma amostra em execuções diferentes.

---

#### **2. Treinamento dos Modelos**

##### **Decision Tree (`train_tree.py`)**

```python
dt = DecisionTreeClassifier(
    max_depth=10,              # Limita profundidade
    min_samples_split=20,      # Mínimo de amostras para split
    min_samples_leaf=10,       # Mínimo de amostras por folha
    random_state=42
)
```

**Decisões de Hiperparâmetros:**

| Parâmetro | Valor | Por Que? | Risco se Diferente |
|-----------|-------|----------|-------------------|
| `max_depth=10` | 10 níveis | Profundidade média para datasets tabulares. Previne overfitting em dados com ruído. | **Muito alto (ex: 50):** Árvore memoriza treino (overfitting). **Muito baixo (ex: 3):** Underfitting, não captura padrões. |
| `min_samples_split=20` | 20 amostras | Evita splits em subconjuntos pequenos (pouco representativos). | **Muito baixo (ex: 2):** Árvore cria regras específicas para poucos exemplos (overfitting). |
| `min_samples_leaf=10` | 10 amostras | Garante que folhas tenham exemplos suficientes para generalizar. | **Muito baixo (ex: 1):** Folhas com 1 exemplo = decorar dataset. |

**Por que Decision Trees são robustas:**
- **Não precisam de escalonamento:** Splits baseados em thresholds (ex: "Temperatura > 25°C?").
- **Lidam com não-linearidade:** Capturam interações complexas (ex: "SE temperatura > 30 E umidade < 40 ENTÃO...").
- **Interpretabilidade:** Regras legíveis por humanos.
- **Overfitting fácil:** Sem regularização, decoram o treino. Por isso os hiperparâmetros acima.

---

##### **KNN (`train_knn.py`)**

```python
knn = KNeighborsClassifier(
    n_neighbors=5,    # K=5 vizinhos
    n_jobs=-1         # Paralelização (usa todos os cores)
)
```

**Seleção do K=5:**
```python
# Testamos K de 1 a 20 e plotamos acurácia
k_range = list(range(1, 21))
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # ... treinar e avaliar
# Gráfico mostra K=5 como melhor trade-off
```

| K | Comportamento | Por Que Não Escolher? |
|---|---------------|----------------------|
| **K=1** | Classifica baseado no vizinho mais próximo | **Overfitting:** Sensível a ruído. Um outlier mislabeled pode quebrar predição. |
| **K=5** | **Balanceado** | Suaviza ruído sem perder granularidade. |
| **K=20** | Classifica baseado em 20 vizinhos | **Underfitting:** Fronteiras de decisão muito suaves. Perde detalhes. |

---

##### **SVM (`train_svm.py`)**

```python
svm = SVC(
    kernel='rbf',        # Radial Basis Function (não-linear)
    C=1.0,               # Regularização
    probability=True,    # Habilita predict_proba() para ROC-AUC
    random_state=42
)
```

**Decisões de Hiperparâmetros:**

| Parâmetro | Valor | Por Que? |
|-----------|-------|----------|
| `kernel='rbf'` | Radial Basis Function | **Não-linear:** Mapeia dados para espaço de alta dimensão onde são linearmente separáveis. Alternativas: `'linear'` (mais rápido, assume separabilidade linear), `'poly'` (polinomial, caro computacionalmente). |
| `C=1.0` | Penalização padrão | **Trade-off:** C alto → Margem estreita (overfitting). C baixo → Margem larga (underfitting). 1.0 é balanceado. |
| `probability=True` | Habilita probabilidades | **Necessário para ROC-AUC:** `predict()` retorna classes (0, 1). `predict_proba()` retorna probabilidades (0.0-1.0). Aumenta tempo de treino (~2x), mas essencial para métricas. |

**PCA Opcional (Redução de Dimensionalidade):**
```python
if use_pca:
    pca = PCA(n_components=2)  # Reduz para 2 dimensões
    X_train_pca = pca.fit_transform(X_train_scaled)
```
- **Por que PCA?** SVM é O(n² a n³) em número de amostras. Com muitas features, treino fica lento. PCA reduz features mantendo variância.
- **Trade-off:** Perda de informação. 2 componentes podem capturar 80% da variância, mas 20% é perdido.

---

#### **3. Métricas de Avaliação (`util_metrics.py`)**

```python
metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred, average='macro'),
    'recall': recall_score(y_true, y_pred, average='macro'),
    'f1_score': f1_score(y_true, y_pred, average='macro'),
    'roc_auc': roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
}
```

**Por que múltiplas métricas?**

| Métrica | O que mede | Quando usar |
|---------|------------|-------------|
| **Acurácia** | % de predições corretas | Classes balanceadas. **Cuidado:** 90% acurácia em dataset com 90% classe A = modelo inútil (prediz sempre A). |
| **Precisão** | % de positivos previstos que são realmente positivos | Falsos positivos são caros. Ex: spam (marcar email legítimo como spam). |
| **Recall** | % de positivos reais que foram identificados | Falsos negativos são caros. Ex: diagnóstico de câncer (não detectar doença). |
| **F1-Score** | Média harmônica de Precisão e Recall | Trade-off entre FP e FN. Classes desbalanceadas. |
| **ROC-AUC** | Área sob curva ROC (varia de 0 a 1) | Avalia performance em diferentes thresholds. 1.0 = perfeito, 0.5 = aleatório. |


---


### **Parte 3: Algoritmo Genético**

#### Execução no terminal:
```bash
make part3
```

#### Decisões de implementação:
- **Problema implementado:** Knapsack 0/1 (Problema da Mochila);
- Foi utilizada a semente aleatória `random.seed(42)` para padronizar a geração de itens;
- A codificação na geração de itens foi feita em uma lista de tuplas, em que o primeiro número da tupla armazena o peso do item e o segundo número armazena o valor do item (_fitness_). Isso também é imprimido no terminal da seguinte forma:
```
[(1, 1), (3, 4), (2, 3), (1, 2), (5, 7), (1, 1), (1, 4), (2, 1), (5, 4), (5, 7)] 
```

- A codificação dos genes é feita de forma binária, com base no índice da lista de itens do problema. No caso, um gene é uma lista do tamanho do número de itens, em que 1 indica que o item foi selecionado para entrar na mochila e 0 indica que o item não foi selecionado, como pode ser visto a seguir:

```
[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
```

- O problema foi parametrizado com base nos seguintes parâmetros e valores padrão para execução:

| Parâmetros | Valores |
|------------|---------|
| Número de itens | 10 |
| Peso máximo por item | 5 |
| Valor máximo por item | 8 |
| Peso máximo da mochila | 10 |
| Tamanho da população | 400 |
| Número de gerações | 1000 |
| Taxa inicial de mutação | 1% |
| Taxa variável de mutação | 5% | 

- O fitness de um gene é calculado com base na soma dos valores dos itens no gene quando a soma dos pesos dos itens do gene é menor ou
igual ao limite de peso da mochila. Caso contrário, o fitness é zerado.

- A implementação conta com operadores característicos do algoritmo genético:
  - _Elitismo:_ Permite no código com que ma que os dois genes com maior fitness de
  uma geração sejam preservados para a próxima geração, desde que ambos possuam fitness maior que zero.
  - _Seleção de Pais por roleta:_ Cada gene recebe uma porcentagem com base em seu fitness, que dita a chance de que ele se torne pai através da roleta viciada (sem contar com o elitismo).
  - _Taxa de mutação variável:_ Após 50 gerações, a taxa de mutação sobe de 1% para 5%, a fim de garantir movimentações bem-vindas na diversidade genética dos genes disponíveis após esse momento.

---

### **Parte 4: Algoritmos de Enxame e Imunes**

#### Execução no terminal:
```bash
# Executar ambos os algoritmos:
make part4

# Executar apenas ACO vs GA:
make part4-aco

# Executar comparação completa (CLONALG vs ACO vs GA):
make part4-clonalg
```

#### O que faz:
Esta parte implementa e compara **três algoritmos bio-inspirados** resolvendo o **Problema da Mochila (Knapsack 0/1)**:

1. **ACO (Ant Colony Optimization)** - Otimização por Colônia de Formigas
2. **GA (Genetic Algorithm)** - Algoritmo Genético (da Parte 3)
3. **CLONALG (Clonal Selection Algorithm)** - Seleção Clonal (Sistema Imune Artificial)

---

#### **1. ACO - Ant Colony Optimization (`aco.py`)**

##### Como funciona:
Imagine formigas procurando comida. Elas deixam **feromônios** no caminho, e outras formigas seguem trilhas com mais feromônio. Com o tempo, o caminho mais curto acumula mais feromônio (porque as formigas completam o trajeto mais rápido).

**No Knapsack:**
- Cada "formiga" constrói uma solução selecionando itens probabilisticamente
- Itens com **maior eficiência (valor/peso)** e **mais feromônio** têm maior chance de serem escolhidos
- Boas soluções reforçam o feromônio nos itens selecionados
- Feromônio evapora com o tempo (evita convergência prematura)

##### Parâmetros principais:

| Parâmetro | Valor | O que faz |
|-----------|-------|-----------|
| `n_ants=30` | 30 formigas | Cada formiga cria uma solução por iteração |
| `alpha=1.0` | Peso do feromônio | Controla influência da trilha de feromônio |
| `beta=2.0` | Peso da heurística | Controla influência da eficiência do item (valor/peso) |
| `rho=0.5` | Taxa de evaporação | 50% do feromônio evapora a cada iteração |
| `Q=100` | Quantidade de feromônio | Quanto feromônio é depositado nas soluções |

##### Decisões de implementação:

**Elitismo no depósito de feromônio:**
```python
# Apenas as top 30% soluções depositam feromônio
n_elite = max(1, int(0.3 * len(solutions)))
for sol, val, _ in solutions_sorted[:n_elite]:
    # Deposita feromônio proporcional ao valor da solução
```
- **Por quê** evita que soluções ruins "poluam" a trilha de feromônio
- **Alternativa:** Todas as formigas depositam (converge muito rápido)

**Heurística baseada em eficiência:**
```python
self.heuristic = np.array([item.efficiency for item in items])
# efficiency = value / weight (quanto valor por unidade de peso)
```
- **Por quê** guia as formigas para itens "valiosos e leves"
- **Impacto:** Converge ~10x mais rápido que sem heurística

**Resultados esperados:**
- Encontra solução **ótima ou próxima do ótimo** (valor ~2239)
- Convergência **rápida** (primeiras 10-20 iterações)
- Alta **diversidade** de soluções (30/30 formigas encontram caminhos diferentes para o mesmo valor)

---

#### **2. CLONALG - Clonal Selection Algorithm (`clonag.py`)**

##### Como funciona:
Inspirado no **sistema imune humano**. Quando um vírus invade o corpo, células de defesa (linfócitos) que reconhecem o invasor se multiplicam (clonagem) e sofrem mutações para melhorar o reconhecimento.

**No Knapsack:**
- **População inicial:** 80 soluções aleatórias
- **Seleção:** Escolhe as 15 melhores soluções (maior valor)
- **Clonagem:** Cada solução selecionada gera clones (quanto melhor a solução, mais clones)
- **Hipermutação:** Clones sofrem mutações (soluções piores mutam mais)
- **Seleção clonal:** Clones competem entre si, sobrevivem os melhores
- **Injeção de diversidade:** 30 soluções aleatórias entram a cada geração

##### Parâmetros principais:

| Parâmetro | Valor | O que faz |
|-----------|-------|-----------|
| `pop_size=80` | 80 indivíduos | Tamanho da população |
| `n_select=15` | 15 selecionados | Quantos serão clonados |
| `beta=1.2` | Fator de clonagem | Controla quantos clones por solução |
| `n_random=30` | 30 aleatórios | Novas soluções injetadas por geração |

##### Decisões de implementação:

**Taxa de mutação adaptativa:**
```python
# Soluções piores mutam mais (exploração)
# Soluções melhores mutam menos (exploração fina)
val_norm = val / (max_val + 1e-9)  # Normaliza entre 0 e 1
rate = 0.5 * (1.0 - val_norm * 0.8)  # 0.1 a 0.5
```
- **Por quê** soluções ruins precisam mudar drasticamente. Soluções boas precisam apenas ajustes finos.
- **Impacto:** Evita convergência prematura (toda população idêntica)

**Hipermutação para indivíduos convergidos:**
```python
# Se uma solução é muito similar à melhor (< 5 bits diferentes)
if hamming_distance(sol, best_sol) < 5:
    # Aplica mutação muito agressiva (40-60%)
    rate = np.random.uniform(0.4, 0.6)
```
- **Por quê** "sacode" a população quando ela está estagnada
- **Problema resolvido:** Versão anterior colapsava para 1/80 soluções únicas. Agora mantém 56-69/80.

**Seleção híbrida (50% fitness + 50% diversidade):**
```python
# Metade da população: melhores por valor
elite = population[:pop_size // 2]

# Outra metade: remove duplicatas exatas
diverse = [soluções únicas]
```
- **Por quê** balanceia qualidade (não perder boas soluções) com diversidade (não decorar)
- **Trade-off:** Pode manter soluções subótimas se forem únicas

**Reparo de soluções inválidas:**
```python
# Se peso excede capacidade, remove itens de menor eficiência
while weight > capacity:
    # Remove item menos eficiente (menor valor/peso)
```
- **Por quê** as mutações podem gerar soluções que excedem a capacidade da mochila
- **Alternativa:** Penalizar com fitness=0 (desperdiça avaliações)

**Resultados esperados:**
- Encontra solução **próxima do ótimo** (valor ~2214, 98.9% do ótimo)
- Convergência **gradual** (melhora ao longo de 100 gerações)
- Boa **diversidade** mantida (56-69/80 soluções únicas)

---

#### **3. Comparação: ACO vs GA vs CLONALG**

##### Gráficos gerados:

**Figura 1 - Convergência e Média:**
- Mostra como cada algoritmo encontra soluções melhores ao longo do tempo
- ACO converge rápido (iteração 1-10), GA e CLONALG gradualmente

**Figura 2 - Diversidade e Taxa de Convergência:**
- **Diversidade:** Quantas soluções diferentes existem na população
  - ACO: ~30/30 (todas diferentes, mesmo valor)
  - CLONALG: ~60/80 (boa diversidade)
  - GA: ~100/100 (máxima diversidade)
- **Taxa de convergência:** Velocidade de melhoria (valor ganho por iteração)

**Figura 3 - Eficiência e Comparação Final:**
- **Eficiência:** Valor final ÷ tempo de execução (quanto "retorno por segundo")
  - CLONALG: ~475 valor/s (mais rápido)
  - ACO: ~230 valor/s
  - GA: ~200 valor/s
- **Qualidade vs Tempo:** Compara valor final e tempo de execução

##### Por que ACO vence?

| Métrica | ACO | GA | CLONALG |
|---------|-----|-----|---------|
| **Valor final** | 2239 (100%) | ~2100 (93.8%) | 2214 (98.9%) |
| **Tempo** | ~10s | ~10s | ~5s |
| **Convergência** | Rápida (10 iter) | Lenta (100 ger) | Média (60 ger) |
| **Diversidade** | Alta (30/30) | Alta (100/100) | Média (60/80) |

**ACO é melhor porque:**
1. **Heurística guiada:** Usa eficiência (valor/peso) como "cheiro" inicial
2. **Exploração inteligente:** Feromônio reforça bons caminhos, mas evaporação permite exploração
3. **Elitismo controlado:** Só 30% depositam feromônio (evita convergência prematura)

**Quando usar cada algoritmo:**
- **ACO:** Problemas com **estrutura de grafo** (caminhos, rotas, sequências). Converge rápido com heurística boa.
- **GA:** Problemas **sem heurística óbvia**. Genérico, funciona para tudo (mas não é o melhor em nada).
- **CLONALG:** Problemas que precisam **alta diversidade** (multiobjetivo, paisagens rugosas). Rápido e eficiente.

---

#### Problema resolvido: Knapsack 0/1

**Configuração padrão:**
- **50 itens** gerados aleatoriamente (peso: 5-50, valor: 10-100)
- **Capacidade da mochila:** 686 (metade do peso total)
- **Semente aleatória:** 42 (itens), 100/200/300 (GA/ACO/CLONALG)

**Solução ótima encontrada pelo ACO:**
- **Valor:** 2239
- **Peso:** 686/686 (100% da capacidade)
- **Itens selecionados:** 32/50

**Por que não testamos algoritmos exatos (programação dinâmica)?**
- Knapsack 0/1 tem solução exata O(n×W), onde n=itens e W=capacidade
- Para n=50, W=686: ~34.300 operações (instantâneo)
- **Mas:** Algoritmos exatos não escalam. Com n=10.000, W=100.000: 1 bilhão de operações
- **Algoritmos bio-inspirados** escalam melhor (tempo proporcional a população×gerações, não a capacidade)

---

## Resultados e Comparações

### Visualizar Resultados

```bash
make results
```

Este comando exibe:
- **Relatório comparativo** (`comparison_report.txt`) com métricas de todos os algoritmos
- **Benchmark CSV** (`benchmark_results.csv`) com dados tabulares
- **Matrizes de confusão** (`.png`) para análise visual de erros
- **Gráfico de seleção de K** (KNN) mostrando por que K=5 foi escolhido
- **Visualização da Decision Tree** (estrutura completa da árvore)

### Interpretação dos Resultados

**Comparação Típica (Water Quality Dataset):**

| Algoritmo | Acurácia Teste | Tempo Treino | Overfitting |
|-----------|----------------|--------------|-------------|
| **Decision Tree** | ~98.7% | ~0.45s | Baixo (0.008%) | 
| **KNN** | ~92.6% | ~0.23s | Médio (2.8%) | 
| **SVM** | ~95.0% | ~590s | Baixo (-0.07%) | 

**Análise:**
- **Decision Tree:** Melhor acurácia e interpretabilidade. Rápida de treinar. **Escolha ideal para este dataset.**
- **KNN:** Treino rápido, mas performance inferior. Overfitting moderado (decora padrões locais do treino).
- **SVM:** Boa acurácia, mas treino MUITO lento (10 minutos para 100k linhas). Modelo "black box" (difícil interpretar).

**Por que Decision Tree venceu aqui?**
1. **Dataset tabular com features numéricas:** Árvores são naturalmente adequadas para dados estruturados.
2. **Interações não-lineares:** Água potável depende de combinações (ex: pH alto + Cloro baixo = não potável).
3. **Dados limpos:** Poucas outliers extremas, então robustez do KNN não foi necessária.
4. **Interpretabilidade requerida:** Podemos extrair regras (ex: "SE Hardness < 200 E Solids > 20000 ENTÃO potável").

---

## Estrutura de Arquivos Gerados

```
data/processed/
├── benchmark_results.csv           # Métricas de todos os treinos (Parte 2)
├── comparison_report.txt           # Relatório formatado (Parte 2)
├── confusion_matrix_dt_100000.png  # Matriz de confusão (Decision Tree)
├── confusion_matrix_knn_100000.png # Matriz de confusão (KNN)
├── confusion_matrix_svm_100000.png # Matriz de confusão (SVM)
├── decision_tree_visualization_100000.png  # Árvore completa
├── knn_k_selection.png             # Gráfico de seleção de K
├── aco_fig1_convergencia_media.png # ACO vs GA: Convergência e média
├── aco_fig2_diversidade_feromonios.png # ACO: Diversidade e feromônios
├── aco_fig3_eficiencia_comparacao.png  # ACO vs GA: Eficiência
├── fig1_convergencia_media.png     # CLONALG vs ACO vs GA: Convergência
├── fig2_diversidade_taxa.png       # CLONALG vs ACO vs GA: Diversidade
├── fig3_eficiencia_comparacao.png  # CLONALG vs ACO vs GA: Eficiência
├── X_train.csv                     # Features de treino (não escalonadas)
├── X_train_scaled.csv              # Features de treino (escalonadas)
├── X_test.csv                      # Features de teste (não escalonadas)
├── X_test_scaled.csv               # Features de teste (escalonadas)
├── y_train.csv                     # Labels de treino
└── y_test.csv                      # Labels de teste
```


---

## Referências 

- **"Introduction to Machine Learning with Python"** - Andreas Müller & Sarah Guido
  - Capítulos 2-3: Pré-processamento e validação
  - Capítulo 5: Decision Trees, KNN, SVM
- **"Hands-On Machine Learning"** - Aurélien Géron
  - Capítulo 6: Decision Trees e Random Forests
  - Capítulo 5: SVM e Kernel Trick

### Documentação Oficial
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [pandas Documentation](https://pandas.pydata.org/docs/)

---

## Licença e Autoria

**Disciplina:** Inteligência Artificial (2025/2)  
**Professor:** Tiago Alves de Oliveira  
**Instituição:** Cefet-MG Divinópolis
**Alunos:** João Pedro Rodrigues Silva e Samuel Silva Gomes

---

## Contato

Para dúvidas ou sugestões, abra uma issue no repositório ou entre em contato com os autores.

