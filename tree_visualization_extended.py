"""
Extensão do módulo tree.py com visualização colorida por categorias filosóficas
"""
import matplotlib.pyplot as plt
import networkx as nx
import textwrap
from collections import deque

def visualize_with_categories(tree, title="Árvore de Decisão Filosófica", figsize=(28, 16), 
                              save_path=None, max_char_per_line=20, show_legend=True, show_explanation=True):
    """
    Visualiza a árvore com cores diferentes para cada categoria filosófica.
    Otimizada para melhor legibilidade - mostra apenas o início dos textos.
    
    Argumentos:
        tree: Árvore Tree a ser visualizada
        title: Título do gráfico
        figsize: Tamanho da figura (padrão: 28x16 para melhor visualização)
        save_path: Caminho para salvar
        max_char_per_line: Caracteres por linha (padrão: 20 para textos curtos)
        show_legend: Mostrar legenda de cores
        show_explanation: Mostrar explicação da estrutura antes da visualização
    """
    
    if show_explanation:
        print("\n" + "=" * 80)
        print("EXPLICAÇÃO DA ESTRUTURA HIERÁRQUICA DAS PERGUNTAS")
        print("=" * 80)
        print("""
NÍVEL 0 - PERGUNTA FUNDAMENTAL (Raiz):
--------------------------------------
"Você acredita que existe uma verdade objetiva e universal?"

Esta é a PRIMEIRA pergunta porque estabelece a divisão mais fundamental:
- SIM → Realismo: Filosofias que acreditam em verdades objetivas
- NÃO → Anti-realismo: Filosofias céticas ou relativistas

NÍVEL 1 - EPISTEMOLOGIA:
------------------------
Para quem respondeu SIM:
  → "A razão é superior à experiência sensorial?"
    - Separa RACIONALISTAS (razão) de EMPIRISTAS (experiência)

Para quem respondeu NÃO:
  → "A existência humana possui algum valor inerente?"
    - Separa quem CRIA significado de NIILISTAS

NÍVEL 2 - METODOLOGIA:
---------------------
Refina as abordagens filosóficas em 4 grandes grupos

NÍVEIS 3-5 - ESPECIFICAÇÃO:
--------------------------
Distingue correntes específicas dentro de cada escola

CÓDIGO DE CORES NO GRAFO:
-------------------------
- AZUL: Racionalismo (razão pura, ideias inatas)
- VERDE: Empirismo/Positivismo (experiência, observação)
- LARANJA: Pragmatismo (utilidade, consequências)
- AMARELO: Hedonismo/Ética (prazer, virtude, felicidade)
- ROXO: Existencialismo (autenticidade, liberdade)
- VERMELHO: Niilismo (ausência de valores objetivos)
- CINZA: Análise Linguística/Outros
        """)
        print("=" * 80 + "\n")
        input("Pressione ENTER para visualizar o grafo colorido...")
    
    def categorizar_node(text):
        """Determina a categoria filosófica baseada no conteúdo do nó"""
        text_upper = text.upper()
        
        if "RACIONALISMO" in text_upper or "PLATONICO" in text_upper or "CARTESIANO" in text_upper or \
           "LEIBNIZ" in text_upper or "SPINOZ" in text_upper or ("RAZAO" in text_upper and "SUPERIOR" in text_upper):
            return "Racionalismo", "#BBDEFB"  # Azul claro
        elif "EMPIRISMO" in text_upper or "POSITIVISMO" in text_upper or "EXPERIENCIA" in text_upper or \
             "LOCKE" in text_upper or "HUME" in text_upper or "BERKELEY" in text_upper:
            return "Empirismo", "#C8E6C9"  # Verde claro
        elif "PRAGMATISMO" in text_upper or "CONVENCIONALISMO" in text_upper or \
             "CONSEQUENCIAS" in text_upper or ("UTILIDADE" in text_upper and "CRITERIO" in text_upper):
            return "Pragmatismo", "#FFCC80"  # Laranja claro
        elif "HEDONISMO" in text_upper or "EPICUR" in text_upper or "UTILITARISMO" in text_upper or \
             "EUDEMON" in text_upper or "ARISTOTEL" in text_upper or ("PRAZER" in text_upper and "SUPREMO" in text_upper) or \
             ("FELICIDADE" in text_upper and "OBJETIVO" in text_upper) or ("VIRTUDE" in text_upper and "IMPORTANTES" in text_upper):
            return "Hedonismo/Etica", "#FFF9C4"  # Amarelo claro
        elif "EXISTENCIALISMO" in text_upper or "SARTRE" in text_upper or "CAMUS" in text_upper or \
             "KIERKEGAARD" in text_upper or "HEIDEGGER" in text_upper or "AUTENTICIDADE" in text_upper or \
             ("LIBERDADE" in text_upper and ("INDIVIDUAL" in text_upper or "ANGUSTIA" in text_upper)):
            return "Existencialismo", "#E1BEE7"  # Roxo claro
        elif "NIILISMO" in text_upper or "NIETZSCHE" in text_upper or ("VALORES" in text_upper and "CONSTRUCOES" in text_upper):
            return "Niilismo", "#FFCDD2"  # Vermelho claro
        elif "LINGUAGEM" in text_upper or "ANALITICA" in text_upper or ("PROBLEMAS" in text_upper and "FILOSOFICOS" in text_upper):
            return "Analise Linguistica", "#CFD8DC"  # Cinza claro
        else:
            return "Questao Geral", "#F5F5F5"  # Cinza muito claro
    
    def wrap_text(text, width=25):
        """
        Extrai apenas o início do texto para exibição nos nós.
        - Para correntes filosóficas: mostra apenas o nome
        - Para perguntas: mostra apenas as primeiras palavras
        """
        text = text.strip()
        
        # Se contém quebra de linha, é um resultado (corrente filosófica)
        if '\n' in text:
            # Pega apenas a primeira linha (nome da corrente)
            nome_corrente = text.split('\n')[0]
            return nome_corrente
        
        # Se for uma pergunta, pega apenas o início
        # Remove "Você acredita que", "A", etc. e pega as primeiras palavras-chave
        palavras = text.split()
        
        # Limita a 6-8 palavras principais
        if len(palavras) <= 6:
            texto_curto = text
        else:
            # Pega até 60 caracteres ou até encontrar um ponto de interrogação
            if '?' in text and text.index('?') < 60:
                texto_curto = text[:text.index('?')+1]
            else:
                texto_curto = ' '.join(palavras[:6]) + '...?'
        
        # Quebra em linhas se necessário
        return '\n'.join(textwrap.wrap(texto_curto, width=width, break_long_words=False))
    
    def build_graph(tree_node, graph, pos, categories, x=0, y=0, layer=1, parent=None, direction=""):
        if tree_node is None:
            return graph, pos, categories
        
        node_id = id(tree_node)
        wrapped_label = wrap_text(tree_node.value, width=max_char_per_line)
        categoria, cor = categorizar_node(tree_node.value)
        
        graph.add_node(node_id, label=wrapped_label, category=categoria, color=cor)
        pos[node_id] = (x, y)
        categories[node_id] = (categoria, cor)
        
        if parent is not None:
            graph.add_edge(parent, node_id, label=direction)
        
        # Espaçamento aumentado para melhor legibilidade
        width = 10 / (2 ** layer)
        
        if tree_node.left:
            build_graph(tree_node.left, graph, pos, categories, x - width, y - 2, layer + 1, node_id, "Sim")
        if tree_node.right:
            build_graph(tree_node.right, graph, pos, categories, x + width, y - 2, layer + 1, node_id, "Não")
        
        return graph, pos, categories
    
    G = nx.DiGraph()
    positions = {}
    node_categories = {}
    build_graph(tree, G, positions, node_categories)
    
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('white')
    
    labels = nx.get_node_attributes(G, 'label')
    
    # Separa nós por categoria para desenhar com cores diferentes
    categories_unique = {}
    for node_id, (cat, cor) in node_categories.items():
        if cat not in categories_unique:
            categories_unique[cat] = {'nodes': [], 'color': cor}
        categories_unique[cat]['nodes'].append(node_id)
    
    # Desenha nós por categoria
    for categoria, data in categories_unique.items():
        node_list = data['nodes']
        node_color = data['color']
        
        # Calcula tamanho dos nós de forma mais uniforme
        node_sizes = []
        for node in node_list:
            label = labels[node]
            num_lines = label.count('\n') + 1
            num_chars = len(label)
            
            # Tamanho baseado em linhas e caracteres para melhor ajuste
            # Nós com textos curtos ficam menores
            if num_chars < 30:
                size = 2500 + (num_lines * 300)
            elif num_chars < 60:
                size = 3000 + (num_lines * 350)
            else:
                size = 3500 + (num_lines * 400)
            
            node_sizes.append(size)
        
        nx.draw_networkx_nodes(
            G, positions,
            nodelist=node_list,
            node_color=node_color,
            node_size=node_sizes,
            node_shape='s',  # Retangular
            edgecolors='#424242',
            linewidths=2.5,
            ax=ax,
            label=categoria
        )
    
    # Desenha arestas com melhor contraste
    nx.draw_networkx_edges(
        G, positions,
        edge_color='#424242',  # Mais escuro para melhor contraste
        arrows=True,
        arrowsize=20,
        width=2,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.08',
        ax=ax
    )
    
    # Desenha labels dos nós com fonte maior e mais legível
    nx.draw_networkx_labels(
        G, positions, labels,
        font_size=9,  # Aumentado de 7 para 9
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
    
    # Adiciona legenda com melhor posicionamento e tamanho
    if show_legend:
        legend = plt.legend(
            loc='upper left',
            bbox_to_anchor=(0.01, 0.99),
            fontsize=11,
            framealpha=0.95,
            edgecolor='#424242',
            fancybox=True,
            shadow=True,
            title='Categorias Filosóficas',
            title_fontsize=12
        )
        legend.get_title().set_fontweight('bold')
    
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()
