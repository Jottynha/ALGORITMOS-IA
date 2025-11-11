from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from preprocess import preprocess_data

print("\n" + "="*70)
print("TREINAMENTO DE ÁRVORE DE DECISÃO")
print("="*70)

input_file = 'data/raw/Watera.csv'

# Pré-processar dados
# Retorna: X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders, X_original
X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders, X_original = preprocess_data(input_file)

# IMPORTANTE: Árvore de Decisão NÃO precisa de escalonamento!
# Vamos carregar os dados NÃO escalonados que foram salvos
print("\n" + "="*70)
print("CARREGANDO DADOS NÃO ESCALONADOS (melhor para Decision Tree)")
print("="*70)
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")

# Criar e treinar a árvore de decisão
print("\n" + "="*70)
print("TREINANDO DECISION TREE")
print("="*70)
clf = DecisionTreeClassifier(
    max_depth=5,           # Profundidade máxima (evita overfitting)
    min_samples_split=20,  # Mínimo de amostras para dividir um nó
    min_samples_leaf=10,   # Mínimo de amostras em uma folha
    random_state=42
)
clf.fit(X_train, y_train)
print("Árvore treinada com sucesso!")
print(f"  - Profundidade da árvore: {clf.get_depth()}")
print(f"  - Número de folhas: {clf.get_n_leaves()}")
print(f"  - Número de nós: {clf.tree_.node_count}")

# Fazer previsões e avaliar
print("\n" + "="*70)
print("AVALIAÇÃO DO MODELO")
print("="*70)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)

print(f"Acurácia TREINO: {acc_train:.4f} ({acc_train*100:.2f}%)")
print(f"Acurácia TESTE:  {acc_test:.4f} ({acc_test*100:.2f}%)")

if acc_train - acc_test > 0.1:
    print("\nPossível OVERFITTING (diferença > 10% entre treino e teste)")
else:
    print("\nModelo generaliza bem!")

# Relatório de classificação
print("\n" + "="*70)
print("RELATÓRIO DE CLASSIFICAÇÃO (Conjunto de TESTE)")
print("="*70)
print(classification_report(y_test, y_pred_test))

# Importância das features
print("\n" + "="*70)
print("IMPORTÂNCIA DAS FEATURES (Top 10)")
print("="*70)
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': clf.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance.head(10).to_string(index=False))

# Visualizar a árvore de decisão
print("\n" + "="*70)
print("VISUALIZANDO ÁRVORE DE DECISÃO")
print("="*70)
plt.figure(figsize=(20, 12))

# Verificar se há classes únicas
unique_classes = sorted(set(y_train) | set(y_test))
class_names = [f'Classe {i}' for i in unique_classes]

plot_tree(
    clf, 
    feature_names=X_train.columns, 
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Árvore de Decisão - Visualização Completa", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('data/processed/decision_tree_visualization.png', dpi=300, bbox_inches='tight')
print("Árvore salva em: data/processed/decision_tree_visualization.png")
plt.show()

print("\n" + "="*70)
print("TREINAMENTO CONCLUÍDO!")
print("="*70)