"""
TREINAMENTO DECISION TREE - BENCHMARK COM MÚLTIPLOS TAMANHOS DE DATASET
=========================================================================
Testa o algoritmo Decision Tree com 10k, 50k e 100k linhas
Salva métricas comparativas em CSV e gera relatorio
"""
import time
import pandas as pd
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

from preprocess import preprocess_data
from util_metrics import calculate_metrics, save_benchmark_results, print_evaluation_summary, plot_confusion_matrix, generate_comparison_report

print("\n" + "="*80)
print("BENCHMARK DECISION TREE - MULTIPLOS TAMANHOS DE DATASET")
print("="*80)

# Dataset de entrada
input_file = 'data/raw/Watera.csv'

# Tamanhos de dataset para testar
dataset_sizes = [10000, 50000, 100000]

# Parâmetros da Decision Tree
max_depth = 10
min_samples_split = 20
min_samples_leaf = 10

for size in dataset_sizes:
    print("\n" + "="*80)
    print(f"TREINANDO DECISION TREE COM {size:,} LINHAS")
    print("="*80)
    
    # Pré-processar dados
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders, X = preprocess_data(
        input_file, 
        sample_size=size
    )
    
    # Decision Tree NÃO precisa de escalonamento
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    
    # Treinar modelo
    print(f"\nTreinando Decision Tree (max_depth={max_depth})...")
    start_time = time.time()
    
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    dt.fit(X_train, y_train)
    
    train_time = time.time() - start_time
    print(f"Treinamento concluido em {train_time:.4f} segundos")
    print(f"Profundidade da arvore: {dt.get_depth()}")
    print(f"Numero de folhas: {dt.get_n_leaves()}")
    
    # Predições
    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)
    
    # Probabilidades para ROC-AUC (tanto treino quanto teste)
    y_prob_train = dt.predict_proba(X_train)
    y_prob_test = dt.predict_proba(X_test)
    
    # Calcular métricas
    metrics_train = calculate_metrics(y_train, y_pred_train, y_prob_train, average='macro')
    metrics_test = calculate_metrics(y_test, y_pred_test, y_prob_test, average='macro')
    
    # Imprimir resumo
    print_evaluation_summary(metrics_train, metrics_test, train_time, f'Decision Tree (max_depth={max_depth}, {size:,} linhas)')
    
    # Importância das features
    print("\nIMPORTANCIA DAS FEATURES (Top 10):")
    print("-" * 60)
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': dt.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance.head(10).to_string(index=False))
    
    # Salvar resultados no CSV de benchmark
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'algorithm': 'Decision Tree',
        'dataset_size': size,
        'train_time': train_time,
        'test_accuracy': metrics_test['accuracy'],
        'test_precision': metrics_test['precision'],
        'test_recall': metrics_test['recall'],
        'test_f1': metrics_test['f1_score'],
        'test_roc_auc': metrics_test['roc_auc'],
        'train_accuracy': metrics_train['accuracy'],
        'overfitting': metrics_train['accuracy'] - metrics_test['accuracy'],
        'hyperparameters': f'max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}'
    }
    
    save_benchmark_results(results)
    
    # Plotar matriz de confusao e arvore apenas para o maior dataset
    if size == dataset_sizes[-1]:
        print("\nGerando matriz de confusao...")
        plot_confusion_matrix(
            y_test, y_pred_test,
            save_path=f'data/processed/confusion_matrix_dt_{size}.png',
            title=f'Matriz de Confusao - Decision Tree (max_depth={max_depth}, {size:,} linhas)'
        )
        
        # Visualizar a arvore de decisao
        print("\n Visualizando arvore de decisao...")
        plt.figure(figsize=(20, 12))
        
        unique_classes = sorted(set(y_train) | set(y_test))
        class_names = [f'Classe {i}' for i in unique_classes]
        
        plot_tree(
            dt,
            feature_names=X_train.columns,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=8
        )
        plt.title(f"Decision Tree - {size:,} linhas (max_depth={max_depth})", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'data/processed/decision_tree_visualization_{size}.png', dpi=300, bbox_inches='tight')
        plt.close()

print("\n" + "="*80)
print("BENCHMARK DECISION TREE CONCLUIDO!")
print("="*80)
generate_comparison_report()

print("\nResultados salvos em:")
print("  - data/processed/benchmark_results.csv")
print("  - data/processed/comparison_report.txt")
print("\n" + "="*80)
