"""
Utilitários para cálculo de métricas e avaliação de modelos
============================================================
Funções comuns para todos os algoritmos de ML:
- Cálculo de métricas (acurácia, precisão, recall, F1)
- ROC-AUC e matriz de confusão
- Salvamento de resultados comparativos
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


def calculate_metrics(y_true, y_pred, y_prob=None, average='macro'):
    """
    Calcula todas as métricas principais de classificação.
    
    Args:
        y_true: Valores reais (ground truth)
        y_pred: Valores preditos
        y_prob: Probabilidades preditas (opcional, para ROC-AUC)
        average: 'macro', 'micro' ou 'weighted' para métricas multi-classe
    
    Returns:
        dict: Dicionário com todas as métricas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # ROC-AUC (apenas se probabilidades estiverem disponíveis)
    if y_prob is not None:
        try:
            # Para multi-classe, usa ovr (one-vs-rest)
            if len(np.unique(y_true)) > 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, 
                                                   multi_class='ovr', 
                                                   average=average)
            else:
                # Para binário, usa apenas probabilidade da classe positiva
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        except:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None
    
    return metrics


def save_benchmark_results(results_dict, output_file='data/processed/benchmark_results.csv'):
    """
    Salva ou atualiza resultados de benchmark em CSV.
    
    Args:
        results_dict: Dicionário com informações do experimento
        output_file: Caminho do arquivo CSV de saída
    
    Example:
        results = {
            'timestamp': '2024-01-15 10:30:00',
            'algorithm': 'Decision Tree',
            'dataset_size': 10000,
            'train_time': 0.5,
            'test_accuracy': 0.85,
            'test_precision': 0.83,
            'test_recall': 0.82,
            'test_f1': 0.82,
            'test_roc_auc': 0.90,
            'train_accuracy': 0.95,
            'overfitting': 0.10
        }
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Criar DataFrame com o novo resultado
    new_row = pd.DataFrame([results_dict])
    
    # Se arquivo já existe, adicionar; senão, criar novo
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    
    # Salvar
    df.to_csv(output_file, index=False)
    print(f"\n✓ Resultado salvo em: {output_file}")


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, title='Matriz de Confusão'):
    """
    Plota matriz de confusão com visualização bonita.
    
    Args:
        y_true: Valores reais
        y_pred: Valores preditos
        class_names: Nomes das classes (opcional, gera automaticamente se None)
        save_path: Caminho para salvar a imagem (opcional)
        title: Título do gráfico
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Se class_names não foi fornecido, gera automaticamente
    if class_names is None:
        unique_classes = sorted(set(y_true) | set(y_pred))
        class_names = [f'Classe {i}' for i in unique_classes]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Contagem'})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Valor Real', fontsize=12)
    plt.xlabel('Valor Predito', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Matriz de confusão salva em: {save_path}")
    
    plt.close()  # Fecha a figura para evitar problemas com display


def print_evaluation_summary(metrics_train, metrics_test, train_time=None, model_name='Modelo'):
    """
    Imprime resumo formatado da avaliação do modelo.
    
    Args:
        metrics_train: Dict com métricas de treino
        metrics_test: Dict com métricas de teste
        train_time: Tempo de treinamento em segundos (opcional)
        model_name: Nome do modelo
    """
    print("\n" + "="*70)
    print(f"RESUMO DA AVALIAÇÃO - {model_name}")
    print("="*70)
    
    if train_time:
        print(f"\nTempo de treinamento: {train_time:.4f} segundos")
    
    print("\nMÉTRICAS DE DESEMPENHO:")
    print("-"*70)
    print(f"{'Métrica':<20} {'TREINO':>12} {'TESTE':>12} {'Diferença':>12}")
    print("-"*70)
    
    for key in ['accuracy', 'precision', 'recall', 'f1_score']:
        if key in metrics_train and key in metrics_test:
            train_val = metrics_train[key]
            test_val = metrics_test[key]
            diff = train_val - test_val
            
            # Formatar nome da métrica
            metric_name = key.replace('_', ' ').title()
            
            print(f"{metric_name:<20} {train_val:>12.4f} {test_val:>12.4f} {diff:>12.4f}")
    
    # ROC-AUC
    if 'roc_auc' in metrics_test and metrics_test['roc_auc'] is not None:
        train_roc = metrics_train.get('roc_auc', None)
        train_roc_str = f"{train_roc:>12.4f}" if train_roc is not None else f"{'N/A':>12}"
        print(f"{'ROC-AUC':<20} {train_roc_str} {metrics_test['roc_auc']:>12.4f} {'':>12}")
    
    print("-"*70)
    
    # Detectar overfitting
    overfitting = metrics_train['accuracy'] - metrics_test['accuracy']
    if overfitting > 0.1:
        print(f"\n ALERTA: Possível OVERFITTING detectado!")
        print(f"   Diferença de acurácia: {overfitting:.2%}")
    elif overfitting < 0:
        print(f"\n✓ Modelo generaliza bem (teste > treino em {abs(overfitting):.2%})")
    else:
        print(f"\n✓ Modelo generaliza bem (diferença < 10%)")
    
    print("="*70)


def generate_comparison_report(csv_file='data/processed/benchmark_results.csv', 
                               output_file='data/processed/comparison_report.txt'):
    """
    Gera relatório comparativo entre todos os experimentos salvos.
    
    Args:
        csv_file: Arquivo CSV com resultados
        output_file: Arquivo de texto para salvar o relatório
    """
    if not os.path.exists(csv_file):
        print(f" Arquivo {csv_file} não encontrado.")
        return
    
    df = pd.read_csv(csv_file)
    
    # Criar relatório
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RELATÓRIO COMPARATIVO DE BENCHMARKS\n")
        f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Agrupar por algoritmo e tamanho de dataset
        f.write("RESULTADOS POR ALGORITMO E TAMANHO DE DATASET\n")
        f.write("-"*80 + "\n\n")
        
        for algo in df['algorithm'].unique():
            f.write(f"\n### {algo} ###\n\n")
            algo_df = df[df['algorithm'] == algo].sort_values('dataset_size')
            
            f.write(f"{'Size':<10} {'Acc Test':>10} {'Prec':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'Time(s)':>10}\n")
            f.write("-"*80 + "\n")
            
            for _, row in algo_df.iterrows():
                roc_auc_val = row.get('test_roc_auc', None)
                if pd.isna(roc_auc_val) or roc_auc_val is None:
                    roc_auc_str = "N/A"
                else:
                    roc_auc_str = f"{roc_auc_val:.4f}"
                
                train_time_val = row.get('train_time', None)
                if pd.isna(train_time_val) or train_time_val is None:
                    train_time_str = "N/A"
                else:
                    train_time_str = f"{train_time_val:.4f}"
                
                f.write(f"{row['dataset_size']:<10} "
                       f"{row['test_accuracy']:>10.4f} "
                       f"{row['test_precision']:>10.4f} "
                       f"{row['test_recall']:>10.4f} "
                       f"{row['test_f1']:>10.4f} "
                       f"{roc_auc_str:>10} "
                       f"{train_time_str:>10}\n")
            
            f.write("\n")
        
        # Ranking
        f.write("\n" + "="*80 + "\n")
        f.write("RANKING POR MÉTRICA (TESTE)\n")
        f.write("="*80 + "\n\n")
        
        for metric in ['test_accuracy', 'test_f1', 'test_precision', 'test_recall']:
            f.write(f"\n### {metric.replace('test_', '').upper()} ###\n")
            top = df.nlargest(5, metric)[['algorithm', 'dataset_size', metric]]
            f.write(top.to_string(index=False))
            f.write("\n")
    
    print(f"\n✓ Relatório comparativo salvo em: {output_file}")
    
    # Também imprimir no console
    with open(output_file, 'r', encoding='utf-8') as f:
        print("\n" + f.read())


if __name__ == "__main__":
    print("Módulo de utilitários de métricas carregado com sucesso!")
    print("\nFunções disponíveis:")
    print("  - calculate_metrics()")
    print("  - save_benchmark_results()")
    print("  - plot_confusion_matrix()")
    print("  - print_evaluation_summary()")
    print("  - generate_comparison_report()")
