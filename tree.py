# Código responsável por modularizar a estrutura de uma árvore de decisão;
# Adotando 'Não' como direito e 'Sim' como esquerdo;
import matplotlib.pyplot as plt
import networkx as nx
class Tree:
    def __init__(self, value=None):
        self.value = value
        self.left = None
        self.right = None
    def add_left(self, value):
        self.left = Tree(value)
    def add_right(self, value):
        self.right = Tree(value)
    def is_leaf(self):
        return (self.left is None and self.right is None)
    def sim(self):
        return self.left
    def nao(self):
        return self.right
    def print(self):
        def build_graph(tree, graph, pos, x=0, y=0, layer=1, parent=None, direction=""): # Criada com auxílio de Inteligência Artificial
            if tree is None:
                return graph, pos
            node_id = id(tree) # - Adiciona o nó atual
            graph.add_node(node_id, label=tree.value)
            pos[node_id] = (x, y) # - Adiciona aresta do pai para o nó atual
            if parent is not None:
                graph.add_edge(parent, node_id, label=direction)
            width = 4 / (2 ** layer) # - Calcula a largura para os filhos (quanto mais profundo, menor o espaçamento)
            if tree.left: # - Adiciona filho esquerdo (Sim)
                build_graph(tree.left, graph, pos, x - width, y - 1, layer + 1, node_id, "Sim")
            if tree.right: # - Adiciona filho direito (Não)
                build_graph(tree.right, graph, pos, x + width, y - 1, layer + 1, node_id, "Não")
            return graph, pos       
        # - Cria o grafo
        G = nx.DiGraph()
        positions = {}
        build_graph(self, G, positions)
        # - Configurações de visualização
        plt.figure(figsize=(12, 8))
        # - Desenha os nós
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_nodes(G, positions, node_color='lightblue', node_size=3000, node_shape='o')
        # -  Desenha as arestas
        nx.draw_networkx_edges(G, positions, edge_color='gray', arrows=True, arrowsize=20, width=2)
        # - Desenha os rótulos dos nós
        nx.draw_networkx_labels(G, positions, labels, font_size=10, font_weight='bold')
        # - Desenha os rótulos das arestas (Sim/Não)
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, positions, edge_labels,font_size=9, font_color='red')
        plt.title("Árvore de Decisão", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
# Exemplo:
if __name__ == "__main__":
    root = Tree("É um animal?")
    root.add_left("É um mamífero?")
    root.add_right("É uma planta?")
    root.sim().add_left("É um cão?")
    root.sim().add_right("É um pássaro?")
    root.nao().add_left("É uma flor?")
    root.nao().add_right("É uma árvore?")    
    root.print()
