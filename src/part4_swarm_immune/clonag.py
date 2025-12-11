# CLONALG (Clonal Selection Algorithm) para o Problema da Mochila
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

sys.path.append(str(Path(__file__).parent.parent / 'part3_ga'))
sys.path.append(str(Path(__file__).parent))
from ga import GA
from aco import Item, GAWrapper, KnapsackACO


class CLONALG:
    def __init__(self, items: List[Item], capacity: int, pop_size: int = 50, 
                 n_select: int = 10, beta: float = 1.5, n_random: int = 5, 
                 generations: int = 100):
        self.items = items
        self.capacity = capacity
        self.pop_size = pop_size
        self.n_select = n_select
        self.beta = beta
        self.n_random = n_random
        self.generations = generations
        self.n_items = len(items)
        self.best_values_history = []
        self.avg_values_history = []

    def _evaluate(self, solution: List[int]) -> Tuple[int, int]:
        value = sum(self.items[i].value for i, bit in enumerate(solution) if bit == 1)
        weight = sum(self.items[i].weight for i, bit in enumerate(solution) if bit == 1)
        return value, weight

    def _repair(self, solution: List[int]) -> List[int]:
        sol = solution[:]
        _, weight = self._evaluate(sol)
        if weight <= self.capacity:
            return sol
        
        included = [(i, self.items[i].efficiency) for i, bit in enumerate(sol) if bit == 1]
        included.sort(key=lambda x: x[1])
        
        for idx, _ in included:
            sol[idx] = 0
            _, weight = self._evaluate(sol)
            if weight <= self.capacity:
                break
        return sol

    def _mutate(self, solution: List[int], rate: float) -> List[int]:
        return [1 - bit if np.random.random() < rate else bit for bit in solution]

    def run(self, verbose: bool = True) -> Tuple[List[int], int, int]:
        start_time = time.time()
        
        population = []
        for _ in range(self.pop_size):
            sol = [np.random.randint(0, 2) for _ in range(self.n_items)]
            sol = self._repair(sol)
            val, wt = self._evaluate(sol)
            population.append((sol, val, wt))
        
        best_solution, best_value, best_weight = None, 0, 0

        for gen in range(self.generations):
            population.sort(key=lambda x: x[1], reverse=True)
            selected = population[:self.n_select]
            
            if selected[0][1] > best_value:
                best_solution, best_value, best_weight = selected[0]
            
            clones = []
            for rank, (sol, val, _) in enumerate(selected, start=1):
                n_clones = max(1, int(self.beta * self.pop_size / rank))
                max_val = selected[0][1]
                
                for _ in range(n_clones):
                    rate = 0.3 * (1.0 - val / (max_val + 1e-9)) if max_val > 0 else 0.3
                    clone = self._mutate(sol[:], rate)
                    clone = self._repair(clone)
                    cv, cw = self._evaluate(clone)
                    clones.append((clone, cv, cw))
            
            randoms = []
            for _ in range(self.n_random):
                sol = [np.random.randint(0, 2) for _ in range(self.n_items)]
                sol = self._repair(sol)
                val, wt = self._evaluate(sol)
                randoms.append((sol, val, wt))
            
            population = (population + clones + randoms)
            population.sort(key=lambda x: x[1], reverse=True)
            population = population[:self.pop_size]
            
            self.best_values_history.append(best_value)
            self.avg_values_history.append(np.mean([val for _, val, _ in population]))
            
            if verbose and (gen + 1) % 20 == 0:
                print(f"Geração {gen+1}/{self.generations} | Melhor: {best_value} | "
                      f"Médio: {np.mean([val for _, val, _ in population]):.1f} | "
                      f"Tempo: {time.time()-start_time:.2f}s")
        
        if verbose:
            print(f"\n{'='*70}\nRESULTADO FINAL CLONALG\n{'='*70}")
            print(f"Valor: {best_value} | Peso: {best_weight}/{self.capacity} "
                  f"({best_weight/self.capacity*100:.1f}%) | Itens: {sum(best_solution)} | "
                  f"Tempo: {time.time()-start_time:.2f}s\n{'='*70}")
        
        solution_indices = [i for i, bit in enumerate(best_solution) if bit == 1]
        return solution_indices, best_value, best_weight


def plot_comparison_all(aco, ga, clonalg):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(aco.best_values_history, 'b-', label='ACO', linewidth=2)
    axes[0].plot(ga.best_values_history, 'r-', label='GA', linewidth=2)
    axes[0].plot(clonalg.best_values_history, 'g-', label='CLONALG', linewidth=2)
    axes[0].set_xlabel('Iteração/Geração')
    axes[0].set_ylabel('Melhor Valor')
    axes[0].set_title('Convergência: ACO vs GA vs CLONALG')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(aco.avg_values_history, 'b--', label='ACO', linewidth=2)
    axes[1].plot(ga.avg_values_history, 'r--', label='GA', linewidth=2)
    axes[1].plot(clonalg.avg_values_history, 'g--', label='CLONALG', linewidth=2)
    axes[1].set_xlabel('Iteração/Geração')
    axes[1].set_ylabel('Valor Médio')
    axes[1].set_title('Evolução do Valor Médio')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/all_algorithms_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    np.random.seed(42)
    n_items = 50
    items = [Item(i, np.random.randint(10, 101), np.random.randint(5, 51), 0) 
             for i in range(n_items)]
    for item in items:
        item.efficiency = item.value / item.weight
    capacity = sum(item.weight for item in items) // 2

    print(f"\n{'='*70}\nCOMPARAÇÃO: ACO vs GA vs CLONALG - PROBLEMA DA MOCHILA\n{'='*70}")
    print(f"Instância: {n_items} itens | Capacidade: {capacity}\n{'-'*70}")

    print("ALGORITMO GENÉTICO (GA)\n" + "-"*70)
    ga = GAWrapper(items, capacity, population_size=100, generations=100)
    ga_sol, ga_val, ga_wt = ga.run(verbose=True)

    print(f"\n{'-'*70}\nANT COLONY OPTIMIZATION (ACO)\n{'-'*70}")
    aco = KnapsackACO(items, capacity, n_ants=30, n_iterations=100)
    aco_sol, aco_val, aco_wt = aco.run(verbose=True)

    print(f"\n{'-'*70}\nCLONAL SELECTION ALGORITHM (CLONALG)\n{'-'*70}")
    clonalg = CLONALG(items, capacity, pop_size=50, n_select=10, generations=100)
    clonalg_sol, clonalg_val, clonalg_wt = clonalg.run(verbose=True)

    print(f"\n{'='*70}\nCOMPARAÇÃO FINAL\n{'='*70}")
    results = [
        ('GA', ga_val, ga_wt, len(ga_sol)),
        ('ACO', aco_val, aco_wt, len(aco_sol)),
        ('CLONALG', clonalg_val, clonalg_wt, len(clonalg_sol))
    ]
    results.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, val, wt, n) in enumerate(results, 1):
        symbol = '🥇' if i == 1 else '🥈' if i == 2 else '🥉'
        print(f"{symbol} {name}: Valor={val:>4} | Peso={wt:>3}/{capacity} ({wt/capacity*100:.1f}%) | Itens={n}")
    
    best_val = results[0][1]
    print(f"\n{'='*70}")
    for name, val, _, _ in results[1:]:
        diff = ((best_val - val) / val * 100) if val > 0 else 0
        print(f"Melhor que {name}: +{diff:.2f}%")
    
    print(f"{'='*70}")
    plot_comparison_all(aco, ga, clonalg)
