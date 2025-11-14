"""
TREINAMENTO SVM - BENCHMARK COM MÚLTIPLOS TAMANHOS DE DATASET
================================================================
Testa o algoritmo SVM com 10k, 50k e 100k linhas
Salva métricas comparativas em CSV e gera relatório
"""
import time
from datetime import datetime
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from preprocess import preprocess_data
from util_metrics import calculate_metrics, save_benchmark_results, print_evaluation_summary, plot_confusion_matrix, generate_comparison_report

print("\n" + "="*80)
print("BENCHMARK SVM - MULTIPLOS TAMANHOS DE DATASET")
print("="*80)

# Dataset de entrada
input_file = 'data/raw/Watera.csv'

# Tamanhos de dataset para testar
dataset_sizes = [10000, 50000, 100000]

# Parâmetros do SVM
kernel = 'rbf'
C = 1.0
use_pca = True
n_components = 2

for size in dataset_sizes:
    print("\n" + "="*80)
    print(f"TREINANDO SVM COM {size:,} LINHAS")
    print("="*80)
    
    # Pré-processar dados
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders, X = preprocess_data(
        input_file, 
        sample_size=size
    )
    
    # Aplicar PCA se configurado
    if use_pca and X_train_scaled.shape[1] > n_components:
        print(f"\nAplicando PCA para reduzir de {X_train_scaled.shape[1]} para {n_components} features...")
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        print(f"Variancia explicada: {pca.explained_variance_ratio_.sum():.2%}")
        X_train_final = X_train_pca
        X_test_final = X_test_pca
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    
    # Treinar modelo
    print(f"\nTreinando SVM (kernel={kernel}, C={C})...")
    start_time = time.time()
    
    svm = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    svm.fit(X_train_final, y_train)
    
    train_time = time.time() - start_time
    print(f"Treinamento concluido em {train_time:.4f} segundos")
    print(f"Numero de vetores de suporte: {svm.n_support_.sum()}")
    
    # Predições
    y_pred_train = svm.predict(X_train_final)
    y_pred_test = svm.predict(X_test_final)
    
    # Probabilidades para ROC-AUC (tanto treino quanto teste)
    y_prob_train = svm.predict_proba(X_train_final)
    y_prob_test = svm.predict_proba(X_test_final)
    
    # Calcular métricas
    metrics_train = calculate_metrics(y_train, y_pred_train, y_prob_train, average='macro')
    metrics_test = calculate_metrics(y_test, y_pred_test, y_prob_test, average='macro')
    
    # Imprimir resumo
    model_name = f'SVM (kernel={kernel}, C={C}, {size:,} linhas)'
    if use_pca:
        model_name += f' + PCA({n_components})'
    print_evaluation_summary(metrics_train, metrics_test, train_time, model_name)
    
    # Salvar resultados no CSV de benchmark
    hyperparams = f'kernel={kernel}, C={C}'
    if use_pca:
        hyperparams += f', PCA={n_components}'
    
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'algorithm': 'SVM',
        'dataset_size': size,
        'train_time': train_time,
        'test_accuracy': metrics_test['accuracy'],
        'test_precision': metrics_test['precision'],
        'test_recall': metrics_test['recall'],
        'test_f1': metrics_test['f1_score'],
        'test_roc_auc': metrics_test['roc_auc'],
        'train_accuracy': metrics_train['accuracy'],
        'overfitting': metrics_train['accuracy'] - metrics_test['accuracy'],
        'hyperparameters': hyperparams
    }
    
    save_benchmark_results(results)
    
    # Plotar matriz de confusão apenas para o maior dataset
    if size == dataset_sizes[-1]:
        print("\nGerando matriz de confusao...")
        plot_confusion_matrix(
            y_test, y_pred_test,
            save_path=f'data/processed/confusion_matrix_svm_{size}.png',
            title=f'Matriz de Confusao - SVM (kernel={kernel}, {size:,} linhas)'
        )

print("\n" + "="*80)
print("BENCHMARK SVM CONCLUIDO!")
print("="*80)
generate_comparison_report()

print("\nResultados salvos em:")
print("  - data/processed/benchmark_results.csv")
print("  - data/processed/comparison_report.txt")
print("\n" + "="*80)
