"""
TREINAMENTO KNN - BENCHMARK COM MÚLTIPLOS TAMANHOS DE DATASET
================================================================
Testa o algoritmo KNN com 10k, 50k e 100k linhas
Salva métricas comparativas em CSV e gera relatório
"""
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

from preprocess import preprocess_data
from util_metrics import calculate_metrics, save_benchmark_results, print_evaluation_summary, plot_confusion_matrix, generate_comparison_report

print("\n" + "="*80)
print("BENCHMARK KNN - MULTIPLOS TAMANHOS DE DATASET")
print("="*80)

# Dataset de entrada
input_file = 'data/raw/Watera.csv'

# Tamanhos de dataset para testar
dataset_sizes = [10000, 50000, 100000]

# Parâmetros do KNN
best_k = 5

for size in dataset_sizes:
    print("\n" + "="*80)
    print(f"TREINANDO KNN COM {size:,} LINHAS")
    print("="*80)
    
    # Pré-processar dados
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders, X = preprocess_data(
        input_file, 
        sample_size=size
    )
    
    # Treinar modelo
    print(f"\nTreinando KNN com k={best_k}...")
    start_time = time.time()
    
    knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    
    train_time = time.time() - start_time
    print(f"Treinamento concluido em {train_time:.4f} segundos")
    
    # Predições
    y_pred_train = knn.predict(X_train_scaled)
    y_pred_test = knn.predict(X_test_scaled)
    
    # Probabilidades para ROC-AUC (tanto treino quanto teste)
    y_prob_train = knn.predict_proba(X_train_scaled)
    y_prob_test = knn.predict_proba(X_test_scaled)
    
    # Calcular métricas
    metrics_train = calculate_metrics(y_train, y_pred_train, y_prob_train, average='macro')
    metrics_test = calculate_metrics(y_test, y_pred_test, y_prob_test, average='macro')
    
    # Imprimir resumo
    print_evaluation_summary(metrics_train, metrics_test, train_time, f'KNN (k={best_k}, {size:,} linhas)')
    
    # Salvar resultados no CSV de benchmark
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'algorithm': 'KNN',
        'dataset_size': size,
        'train_time': train_time,
        'test_accuracy': metrics_test['accuracy'],
        'test_precision': metrics_test['precision'],
        'test_recall': metrics_test['recall'],
        'test_f1': metrics_test['f1_score'],
        'test_roc_auc': metrics_test['roc_auc'],
        'train_accuracy': metrics_train['accuracy'],
        'overfitting': metrics_train['accuracy'] - metrics_test['accuracy'],
        'hyperparameters': f'k={best_k}'
    }
    
    save_benchmark_results(results)
    
    # Plotar matriz de confusão apenas para o maior dataset
    if size == dataset_sizes[-1]:
        print("\nGerando matriz de confusao...")
        plot_confusion_matrix(
            y_test, y_pred_test,
            save_path=f'data/processed/confusion_matrix_knn_{size}.png',
            title=f'Matriz de Confusao - KNN (k={best_k}, {size:,} linhas)'
        )

print("\n" + "="*80)
print("BENCHMARK KNN CONCLUIDO!")
print("="*80)
generate_comparison_report()

print("\nResultados salvos em:")
print("  - data/processed/benchmark_results.csv")
print("  - data/processed/comparison_report.txt")
print("\n" + "="*80)
