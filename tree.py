# Código responsável por modularizar a estrutura de uma árvore de decisão;
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

class Tree:
    """
    Classe que representa um nó de uma árvore de decisão binária;
    Adotando 'Não' como direito e 'Sim' como esquerdo;
    """
    def __init__(self, value=None):
        self.value = value
        self.left = None
        self.right = None
    def add_left(self, value):
        """Adiciona um filho à esquerda (Sim)"""
        if isinstance(value, Tree):
            self.left = value
        else:
            self.left = Tree(value)
        return self.left
    def add_right(self, value):
        """Adiciona um filho à direita (Não)"""
        if isinstance(value, Tree):
            self.right = value
        else:
            self.right = Tree(value)
        return self.right
    def is_leaf(self):
        """Verifica se o nó é uma folha (não tem filhos)"""
        return (self.left is None and self.right is None)
    def sim(self):
        """Retorna o filho esquerdo (Sim)"""
        return self.left
    def nao(self):
        """Retorna o filho direito (Não)"""
        return self.right
    def from_list(nodes):
        """
        Cria uma árvore a partir de uma lista em ordem de largura (BFS).
        None representa ausência de nó.
        
        Exemplo:
            nodes = ["Raiz", "Sim1", "Não1", "Sim2", None, "Não2", None]
            Cria:
                    Raiz
                   /    \
                Sim1    Não1
                /         \
              Sim2        Não2
        """
        if not nodes or nodes[0] is None:
            return None
        root = Tree(nodes[0])
        queue = deque([root])
        i = 1
        while queue and i < len(nodes):
            current = queue.popleft()
            # Adicionando filho esquerdo (Sim)
            if i < len(nodes) and nodes[i] is not None:
                current.left = Tree(nodes[i])
                queue.append(current.left)
            i += 1
            # Adicionando filho direito (Não)
            if i < len(nodes) and nodes[i] is not None:
                current.right = Tree(nodes[i])
                queue.append(current.right)
            i += 1
        return root
    def to_list(self):
        """
        Converte a árvore para uma lista em ordem de largura (BFS);
        Inclui None para posições vazias;
        Precisa salvar/restaurar a estrutura exata da árvore.
        """
        if not self.value:
            return []
        result = []
        queue = deque([self])
        while queue:
            current = queue.popleft()
            if current:
                result.append(current.value)
                queue.append(current.left)
                queue.append(current.right)
            else:
                result.append(None)
        # Remove None's do final
        while result and result[-1] is None:
            result.pop()
        return result
    def traverse_bfs(self):
        """
        Retorna lista de valores em ordem de largura (BFS);
        Precisa apenas listar/processar os nós existentes;
        """
        if not self.value:
            return []
        result = []
        queue = deque([self])
        while queue:
            current = queue.popleft()
            result.append(current.value)
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
        return result
    def traverse_dfs_preorder(self):
        """
        Retorna lista de valores em pré-ordem (DFS)
        """
        result = []
        if self.value:
            result.append(self.value)
            if self.left:
                result.extend(self.left.traverse_dfs_preorder())
            if self.right:
                result.extend(self.right.traverse_dfs_preorder())
        return result
    def get_height(self):
        """
        Retorna a altura da árvore
        """
        if self.is_leaf():
            return 0
        left_height = self.left.get_height() if self.left else 0
        right_height = self.right.get_height() if self.right else 0
        return 1 + max(left_height, right_height)
    def get_node_count(self):
        """
        Retorna o número total de nós
        """
        count = 1
        if self.left:
            count += self.left.get_node_count()
        if self.right:
            count += self.right.get_node_count()
        return count
    def find_node(self, value):
        """
        Busca um nó pelo valor (BFS)
        """
        queue = deque([self])
        while queue:
            current = queue.popleft()
            if current.value == value:
                return current
            if current.left:
                queue.append(current.left)
            if current.right:
                queue.append(current.right)
        return None
    def visualize(self, title="Árvore de Decisão", figsize=(20, 12), save_path=None, max_char_per_line=30):
        """
        Visualiza a árvore de decisão usando matplotlib e networkx.
        Otimizada para textos longos (perguntas).
        
        Argumentos:
            title: Título do gráfico
            figsize: Tamanho da figura (largura, altura) - padrão aumentado para (20, 12)
            save_path: Caminho para salvar a imagem (opcional)
            max_char_per_line: Máximo de caracteres por linha antes de quebrar (padrão 30)
        """
        import textwrap
        
        def truncate_text(text, max_length=40):
            """Trunca texto longo e adiciona reticências"""
            if len(text) <= max_length:
                return text
            return text[:max_length-3] + "..."
        
        def wrap_text(text, width=30):
            """Quebra texto em múltiplas linhas"""
            # Remove quebras de linha existentes e texto muito longo
            text = text.replace('\n', ' ').strip()
            if len(text) > 150:  # Limita texto muito longo
                text = text[:150] + "..."
            return '\n'.join(textwrap.wrap(text, width=width))
        
        def build_graph(tree, graph, pos, x=0, y=0, layer=1, parent=None, direction=""):
            if tree is None:
                return graph, pos
            
            node_id = id(tree)
            # Aplica quebra de linha no texto
            wrapped_label = wrap_text(tree.value, width=max_char_per_line)
            graph.add_node(node_id, label=wrapped_label)
            pos[node_id] = (x, y)
            
            if parent is not None:
                graph.add_edge(parent, node_id, label=direction)
            
            # Ajusta o espaçamento horizontal baseado na profundidade
            width = 8 / (2 ** layer)
            
            if tree.left:
                build_graph(tree.left, graph, pos, x - width, y - 1.5, layer + 1, node_id, "Sim")
            if tree.right:
                build_graph(tree.right, graph, pos, x + width, y - 1.5, layer + 1, node_id, "Não")
            
            return graph, pos
        
        G = nx.DiGraph()
        positions = {}
        build_graph(self, G, positions)
        
        # Cria figura com fundo branco
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('white')
        
        labels = nx.get_node_attributes(G, 'label')
        
        # Calcula tamanho dos nós baseado no conteúdo
        node_sizes = []
        for node in G.nodes():
            label = labels[node]
            num_lines = label.count('\n') + 1
            # Tamanho base maior e melhor proporção
            size = 3500 + (num_lines * 400)
            node_sizes.append(size)
        
        # Desenha nós com cor gradiente baseado na profundidade
        nx.draw_networkx_nodes(
            G, positions, 
            node_color='#E3F2FD',  # Azul claro
            node_size=node_sizes,
            node_shape='s',  # Formato retangular para acomodar texto
            edgecolors='#1976D2',  # Borda azul escura
            linewidths=2.5,
            ax=ax
        )
        
        # Desenha arestas
        nx.draw_networkx_edges(
            G, positions, 
            edge_color='#424242',  # Cinza mais escuro para melhor contraste
            arrows=True, 
            arrowsize=20, 
            width=2,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.08',  # Leve curvatura
            ax=ax
        )
        
        # Desenha labels dos nós
        nx.draw_networkx_labels(
            G, positions, labels, 
            font_size=9,  # Aumentado para melhor legibilidade
            font_weight='normal',
            font_family='sans-serif',
            verticalalignment='center',
            horizontalalignment='center',
            ax=ax
        )
        
        # Desenha labels das arestas (Sim/Não)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(
            G, positions, edge_labels, 
            font_size=11,  # Aumentado
            font_color='#C62828',  # Vermelho mais escuro
            font_weight='bold',
            ax=ax
        )
        
        plt.title(title, fontsize=20, fontweight='bold', pad=25)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()
    def print(self):
        """
        Alias para visualize()
        """
        self.visualize()
    def __str__(self):
        """
        Representação em string da árvore
        """
        return f"Árvore(valores={self.value}, altura={self.get_height()}, nós={self.get_node_count()})"
    def __repr__(self):
        return self.__str__()
    
# Exemplo de uso:
if __name__ == "__main__":
    print("=" * 60)
    print("EXEMPLO 1: Construção manual (método original)")
    print("=" * 60)
    root = Tree("É um animal?")
    root.add_left("É um mamífero?")
    root.add_right("É uma planta?")
    root.sim().add_left("É um cão?")
    root.sim().add_right("É um pássaro?")
    root.nao().add_left("É uma flor?")
    root.nao().add_right("É uma árvore?")
    print(f"Informações da árvore: {root}")
    print(f"Travessia BFS: {root.traverse_bfs()}")
    print(f"Travessia DFS: {root.traverse_dfs_preorder()}")
    print(f"Lista (formato vetor): {root.to_list()}")
    root.visualize(title="Exemplo 1: Construção Manual")
    print("\n" + "=" * 60)
    print("EXEMPLO 2: Construção a partir de vetor (BFS)")
    print("=" * 60)
    # Criando a mesma árvore usando lista
    nodes = [
        "É um animal?",
        "É um mamífero?", "É uma planta?",
        "É um cão?", "É um pássaro?", "É uma flor?", "É uma árvore?"
    ]
    root2 = Tree.from_list(nodes)
    print(f"Informações da árvore: {root2}")
    print(f"Travessia BFS: {root2.traverse_bfs()}")
    root2.visualize(title="Exemplo 2: Construção por Vetor")
    print("\n" + "=" * 60)
    print("EXEMPLO 3: Árvore com nós ausentes")
    print("=" * 60)
    # Árvore parcial com None indicando ausência de nós
    nodes_partial = [
        "Tem patas?",
        "Quantas patas?", None,  # Sem filho direito no nível 1
        "4 patas", "2 patas"      # Filhos do "Quantas patas?"
    ]
    root3 = Tree.from_list(nodes_partial)
    print(f"Informações da árvore: {root3}")
    print(f"Travessia BFS: {root3.traverse_bfs()}")
    root3.visualize(title="Exemplo 3: Árvore Parcial")
