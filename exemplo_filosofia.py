"""
Exemplo de Árvore de Decisão Filosófica
=================================================
Este exemplo utiliza o módulo tree.py para criar uma árvore de decisão interativa
que identifica qual corrente filosófica melhor se alinha com suas respostas.
=================================================
Árvore com 6 níveis de profundidade e 32 correntes filosóficas possíveis.
Correntes filosóficas abordadas:
- Racionalismo Platônico, Racionalismo Cartesiano, Racionalismo Spinozista, Racionalismo Leibniziano
- Empirismo Lockeano, Empirismo Humeano, Empirismo Berkeleyano, Positivismo Lógico
- Pragmatismo Clássico, Pragmatismo Instrumentalista, Neopragmatismo, Pragmatismo Radical
- Humanismo Secular, Humanismo Renascentista, Humanismo Cívico, Humanismo Existencial
- Existencialismo Sartreano, Existencialismo Camusiano, Existencialismo Kierkegaardiano, Existencialismo Heideggeriano
- Estoicismo Romano, Estoicismo Grego, Cinismo, Estoicismo Moderno
- Niilismo Ativo, Niilismo Passivo, Niilismo Moral, Niilismo Epistemológico
- Hedonismo Cirenaico, Epicurismo, Utilitarismo Hedonista, Eudemonismo
=================================================
Outra Árvore com 6 níveis de profundidade e 32 correntes filosóficas possíveis.
"""

from tree import Tree
def criar_arvore_filosofia():
    """
    Cria a árvore de decisão para identificar tendências filosóficas.
    Total de 32 resultados possíveis (2^5 = 32 folhas).
    """
    # Definindo a árvore usando o método from_list
    # None indica ausência de nó
    nodes = [
        # Nível 0 (Raiz)
        "Você acredita que existe uma verdade objetiva e universal?",
        # Nível 1 (2 nós)
        "A razão é superior à experiência sensorial como fonte de conhecimento?",  # Sim
        "A existência humana possui algum valor ou significado inerente?",  # Não
        # Nível 2 (4 nós)
        "O conhecimento verdadeiro pode ser alcançado pela intuição intelectual?",  # Sim/Sim
        "A verdade de uma ideia depende de suas consequências práticas?",  # Sim/Não
        "A felicidade e o bem-estar são objetivos legítimos da filosofia?",  # Não/Sim
        "A liberdade individual é mais importante que o bem coletivo?",  # Não/Não
        # Nível 3 (8 nós)
        "As Formas ou Ideias platônicas existem independentemente da mente?",  # Sim/Sim/Sim
        "Todo conhecimento deriva da experiência, mesmo que processado pela razão?",  # Sim/Sim/Não
        "A utilidade é o único critério válido para avaliar teorias?",  # Sim/Não/Sim
        "A linguagem e a análise lógica são centrais para a filosofia?",  # Sim/Não/Não
        "O prazer é o bem supremo e deve ser maximizado?",  # Não/Sim/Sim
        "A virtude e o dever são mais importantes que a felicidade pessoal?",  # Não/Sim/Não
        "A autenticidade e a escolha pessoal definem a existência?",  # Não/Não/Sim
        "Todos os valores e significados são construções arbitrárias?",  # Não/Não/Não
        # Nível 4 (16 nós)
        "O mundo sensível é apenas sombra de uma realidade superior?",  # Sim/Sim/Sim/Sim
        "A dúvida metódica é o caminho para a certeza?",  # Sim/Sim/Sim/Não
        "A mente já possui conhecimento inato ao nascer?",  # Sim/Sim/Não/Sim
        "A percepção sensorial é a única fonte de ideias válidas?",  # Sim/Sim/Não/Não
        "As teorias científicas são apenas instrumentos úteis, não verdades?",  # Sim/Não/Sim/Sim
        "A verdade é relativa ao contexto e à comunidade?",  # Sim/Não/Sim/Não
        "A verificação empírica é essencial para proposições significativas?",  # Sim/Não/Não/Sim
        "Os problemas filosóficos são pseudoproblemas linguísticos?",  # Sim/Não/Não/Não
        "O prazer imediato é preferível ao prazer calculado?",  # Não/Sim/Sim/Sim
        "A ataraxia (tranquilidade) é alcançada pelo prazer moderado?",  # Não/Sim/Sim/Não
        "Maximizar o prazer coletivo justifica sacrifícios individuais?",  # Não/Sim/Não/Sim
        "A virtude por si mesma traz felicidade verdadeira?",  # Não/Sim/Não/Não
        "A angústia da liberdade é inseparável da condição humana?",  # Não/Não/Sim/Sim
        "O absurdo da existência deve ser reconhecido e abraçado?",  # Não/Não/Sim/Não
        "A morte de Deus liberta o indivíduo para criar valores próprios?",  # Não/Não/Não/Sim
        "A busca por significado é fútil e deve ser abandonada?",  # Não/Não/Não/Não
        # Nível 5 (32 nós - Folhas/Resultados)
        "RACIONALISMO PLATONICO\nVocê acredita nas Formas eternas e imutáveis como fundamento da realidade. O conhecimento verdadeiro vem da contemplação intelectual das Ideias perfeitas, não do mundo sensível imperfeito.",
        "RACIONALISMO CARTESIANO\nVocê valoriza a dúvida metódica e a certeza do cogito. A razão clara e distinta é o caminho para o conhecimento, começando pela certeza da própria existência pensante.",
        "RACIONALISMO SPINOZISTA\nVocê tende ao monismo e ao determinismo. A mente possui conhecimento inato e tudo segue necessariamente da natureza de Deus/Natureza. A liberdade está em compreender essa necessidade.",
        "EMPIRISMO LOCKEANO\nVocê vê a mente como uma 'tábula rasa'. Todo conhecimento vem da experiência sensorial, processada pela reflexão. Não existem ideias inatas, apenas experiências acumuladas.",
        "PRAGMATISMO INSTRUMENTALISTA\nVocê trata teorias como ferramentas, não verdades. O valor de uma teoria está em sua utilidade prática, não em sua correspondência com a realidade. A ciência é instrumental.",
        "PRAGMATISMO CLASSICO\nVocê avalia ideias por suas consequências práticas. A verdade é relativa à comunidade de investigadores e muda com novas experiências. O conhecimento é falível e revisável.",
        "POSITIVISMO LOGICO\nVocê restringe o conhecimento ao verificável empiricamente. Proposições metafísicas são sem sentido. A filosofia deve esclarecer a linguagem científica através da análise lógica.",
        "NEOPRAGMATISMO\nVocê vê problemas filosóficos como confusões linguísticas. A filosofia deve abandonar a busca por fundamentos e focar na utilidade das descrições. A linguagem cria realidades, não as espelha.",
        "HEDONISMO CIRENAICO\nVocê busca o prazer imediato e intenso. O presente é tudo que importa; o prazer físico direto é o bem supremo. O futuro é incerto, então maximize o prazer agora.",
        "EPICURISMO\nVocê busca a ataraxia através do prazer calculado e moderado. Evite dores, cultive amizades, viva simplesmente. O prazer é ausência de perturbação, não excesso de estímulos.",
        "UTILITARISMO HEDONISTA\nVocê busca maximizar o prazer total na sociedade. O bem moral é aquilo que produz maior felicidade para o maior número. Consequências agregadas importam mais que intenções.",
        "EUDEMONISMO\nVocê identifica felicidade com virtude e excelência. A vida boa vem de cultivar o caráter e realizar o potencial humano. A virtude é sua própria recompensa e traz satisfação profunda.",
        "EXISTENCIALISMO SARTREANO\nVocê acredita que a existência precede a essência. Estamos 'condenados a ser livres' e totalmente responsáveis por nossas escolhas. A angústia da liberdade é inevitável.",
        "EXISTENCIALISMO CAMUSIANO\nVocê reconhece o absurdo da existência - a tensão entre busca de sentido e silêncio do universo. A resposta é a revolta: viver plenamente apesar da absurdidade.",
        "NIILISMO ATIVO\nVocê rejeita valores tradicionais para criar novos valores autênticos. A 'morte de Deus' é libertadora. O sobre-humano cria significado através da vontade de potência.",
        "NIILISMO PASSIVO\nVocê vê toda busca por significado como fútil. Valores, propósitos e verdades são ilusões. A existência é vazia e qualquer tentativa de preencher esse vazio é autoengano.",
        # Continuação nível 5 (mais 16 folhas)
        "RACIONALISMO LEIBNIZIANO\nVocê acredita em verdades de razão inatas e na harmonia preestabelecida. O mundo é o melhor possível, governado por razão suficiente. Mônadas refletem o universo racionalmente.",
        "EMPIRISMO HUMEANO\nVocê é cético quanto à causalidade e à substância. O conhecimento vem de impressões sensoriais. Crenças sobre o futuro baseiam-se em hábito, não em razão necessária.",
        "EMPIRISMO BERKELEYANO\nVocê adota o imaterialismo: 'ser é ser percebido'. Só existem mentes e ideias. A matéria é uma abstração desnecessária. Deus garante a continuidade das percepções.",
        "POSITIVISMO COMTEANO\nVocê vê a ciência como estágio final do conhecimento. A metafísica e religião são superadas. O conhecimento deve ser observável e útil para o progresso social.",
        "PRAGMATISMO RADICAL\nVocê leva o empirismo ao extremo: até as relações são experienciáveis. A realidade é pluralista e em fluxo. A experiência pura precede distinções sujeito-objeto.",
        "CONVENCIONALISMO\nVocê vê teorias científicas como convenções úteis, não descobertas. Escolhemos frameworks por simplicidade e conveniência. A verdade é pragmática, não correspondencial.",
        "FILOSOFIA ANALITICA\nVocê foca na análise lógica da linguagem. Problemas filosóficos surgem de mal-entendidos linguísticos. Clareza conceitual e rigor lógico são essenciais.",
        "FILOSOFIA DA LINGUAGEM ORDINARIA\nVocê examina o uso comum da linguagem para dissolver confusões filosóficas. O significado é uso. Jogos de linguagem variam por contexto.",
        "HEDONISMO PSICOLOGICO\nVocê acredita que todos agem buscando prazer. O hedonismo é descritivo: humanos são naturalmente motivados por prazer e evitam dor. Moralidade deve reconhecer isso.",
        "EPICURISMO MODERNO\nVocê busca prazeres duradouros e sustentáveis. Qualidade sobre quantidade. A ciência moderna confirma que moderação e contemplação trazem bem-estar.",
        "CONSEQUENCIALISMO\nVocê julga ações por resultados totais. Bem-estar agregado importa. Regras morais são guias práticos, não absolutos. Maximizar o bem geral é imperativo.",
        "ARISTOTELISMO\nVocê busca eudaimonia através da virtude prática. O meio-termo é excelência. Realizar a função própria humana (razão prática) traz florescimento.",
        "EXISTENCIALISMO KIERKEGAARDIANO\nVocê enfatiza subjetividade e 'salto de fé'. A angústia vem de escolhas sem garantias. A existência autêntica requer comprometimento apesar da incerteza.",
        "EXISTENCIALISMO HEIDEGGERIANO\nVocê investiga o 'ser-no-mundo'. A autenticidade vem de confrontar a finitude (ser-para-morte). A existência cotidiana é inautêntica; a angústia revela possibilidades genuínas.",
        "NIILISMO MORAL\nVocê nega valores morais objetivos. Bem e mal são construções sem fundamento. A moralidade é ilusão útil ou controle social. Não existem deveres reais.",
        "NIILISMO EPISTEMOLOGICO\nVocê duvida da possibilidade de conhecimento verdadeiro. Todas crenças são igualmente infundadas. Verdade objetiva é inacessível ou inexistente.",
    ]
    return Tree.from_list(nodes)
def criar_arvore_filosofia_2():
    """
    Cria a árvore de decisão do fluxograma.
    Cada nó tem filho esquerdo = "Sim" e filho direito = "Não".
    None indica ausência de nó naquela posição.
    """
    nodes = [
        # Nível 0 (Raiz)
        "A vida tem sentido?",
        # Nível 1
        "O sentido vem de uma força maior?",                    # Sim (índice 1)
        "O sentido da vida pode ser criado?",                    # Não (índice 2)
        # Nível 2
        "A fé é mais importante que a razão para entender o sentido da vida?",  # Sim/Sim (3)
        "A moral é guiada por princípios ao invés de consequências práticas?",  # Sim/Não (4)
        "O sentido é algo criado por cada um?",                              # Não/Sim (5)
        "A falta de sentido leva ao completo vazio?",                         # Não/Não (6)
        # Nível 3
        "Teísmo/Escolástica",                                               # 7 (filho esquerdo de 3)
        "A razão humana é suficiente para entender o sentido da vida?",     # 8 (filho direito de 3)
        "Kantianismo",                                                       # 9 (filho esquerdo de 4)
        "O bem está em alcançar a virtude e o equilíbrio acima de seguir regras fixas?",  #10 (filho direito de 4)
        "A liberdade e a autenticidade são os valores supremos?",            #11 (filho esquerdo de 5)
        "O bem é avaliado com base na felicidade coletiva?",                 #12 (filho direito de 5)
        "Niilismo",                                                          #13 (filho esquerdo de 6)
        "Mesmo assim, é possível criar valores por pura vontade individual?",#14 (filho direito de 6)
        # Nível 4
        None, None,  # filhos de 7 (Teísmo é folha)
        "Deísmo", "Idealismo",  # filhos de 8 (razão suficiente? -> Deísmo / Idealismo)
        None, None,  # filhos de 9 (Kantianismo é folha)
        "Ética das Virtudes", "Humanismo Clássico",  # filhos de 10
        "Existencialismo", "Humanismo Secular",      # filhos de 11
        "Utilitarismo",                              # filho esquerdo de 12
        "O indivíduo deve tentar transformar o mundo pela ação social?",  # filho direito de 12
        None, None,  # filhos de 13 (Niilismo é folha)
        "Nietzsche", "Devemos então suspender o sentido?",  # filhos de 14
        # Nível 5 (folhas e próximos ramos)
        # filhos de "Deísmo" e "Idealismo" (7.. indices já ocupados acima)
        # (manter None quando não houver)
        # preenchendo posições na ordem de nível para manter estrutura legível
        # (continuando em sequência)
        # filhos de "O indivíduo deve tentar transformar..." (marxismo/pragmatismo)
        None, None, None, None, None, None, None, None,  # espaços reservados para alinhamento
        "Marxismo", "Pragmatismo",  # filhos (Sim/Não) de "O indivíduo deve tentar transformar..."
        None, None,  # possíveis posições vazias
        "Ceticismo", "Tudo o que existe é material?",  # filhos de "Devemos então suspender o sentido?"
        # nível final: filhos de "Tudo o que existe é material?"
        "Empirismo", "Idealismo",  # se "Tudo o que existe é material?" => Sim: Empirismo; Não: Idealismo
    ]

    # Ajuste final: caso a sua implementação de Tree.from_list espere uma lista de tamanho específico
    # com None onde não há nós, este `nodes` já usa None para indicar ausências. 
    # Se for necessário, acrescente mais Nones ao final para igualar comprimento fixo.
    return Tree.from_list(nodes)


def fazer_questionario_interativo(arvore):
    """
    Percorre a árvore interativamente, 
    fazendo perguntas e navegando baseado nas respostas do usuário.
    """
    print("\n" + "=" * 80)
    print("DESCUBRA SUA CORRENTE FILOSOFICA - ANALISE DETALHADA")
    print("=" * 80)
    print("\nResponda as perguntas com 'sim' ou 'não' (ou 's'/'n')")
    print("Digite 'sair' a qualquer momento para encerrar.")
    print("Digite 'voltar' para retornar à pergunta anterior.\n")
    
    nodo_atual = arvore
    caminho = []
    historico_nos = [arvore]  # Pilha para permitir voltar
    nivel = 0
    while not nodo_atual.is_leaf():
        print("-" * 80)
        print(f"\n[Nivel {nivel + 1}/6] {nodo_atual.value}")
        while True:
            resposta = input("\n>>> Sua resposta (sim/não/voltar): ").strip().lower()
            if resposta in ['sair', 'exit', 'quit', 'q']:
                print("\nEncerrando o questionario. Ate logo!\n")
                return None
            if resposta in ['voltar', 'back', 'b'] and len(historico_nos) > 1:
                # Remove o nó atual e volta para o anterior
                historico_nos.pop()
                nodo_atual = historico_nos[-1]
                caminho.pop()
                nivel -= 1
                print(f"\n[Voltando para nivel {nivel + 1}]")
                break
            elif resposta in ['voltar', 'back', 'b']:
                print("\nVoce ja esta na primeira pergunta!")
                continue
            if resposta in ['sim', 's', 'yes', 'y']:
                caminho.append("Sim")
                nodo_atual = nodo_atual.sim()
                historico_nos.append(nodo_atual)
                nivel += 1
                break
            elif resposta in ['não', 'nao', 'n', 'no']:
                caminho.append("Não")
                nodo_atual = nodo_atual.nao()
                historico_nos.append(nodo_atual)
                nivel += 1
                break
            else:
                print("Por favor, responda com 'sim', 'não' ou 'voltar'")
    # Chegou em uma folha = resultado final
    print("\n" + "=" * 80)
    print("RESULTADO DA ANALISE FILOSOFICA")
    print("=" * 80)
    print(f"\n{nodo_atual.value}\n")
    print("-" * 80)
    print("Caminho percorrido:", " -> ".join(caminho))
    print(f"Total de decisoes: {len(caminho)}")
    print("=" * 80 + "\n")
    return nodo_atual.value


def mostrar_arvore_completa(arvore):
    """
    Mostra a visualização gráfica completa da árvore de decisão.
    """
    # Necessita de ajuste.
    print("\nGerando visualizacao grafica da arvore de decisao...\n")
    arvore.visualize(
        title="Arvore de Decisao Filosofica Complexa - 32 Correntes",
        figsize=(20, 12)
    )


def mostrar_info_arvore(arvore):
    """
    Exibe informações sobre a estrutura da árvore.
    """
    print("\n" + "=" * 80)
    print("INFORMACOES DA ESTRUTURA DA ARVORE")
    print("=" * 80)
    print(f"Altura da arvore: {arvore.get_height()} niveis")
    print(f"Total de nos: {arvore.get_node_count()}")
    print(f"Total de perguntas (nos internos): {arvore.get_node_count() - 32}")
    print(f"Total de resultados possiveis: 32 correntes filosoficas")
    print(f"Profundidade maxima: 6 niveis de decisao")
    print(f"Complexidade: 2^5 = 32 caminhos possiveis")
    print("\nEstrutura em lista (BFS - primeiros elementos):")
    lista_completa = arvore.to_list()
    print(f"  Raiz: {lista_completa[0][:50]}...")
    print(f"  Total de elementos na lista: {len(lista_completa)}")
    print("=" * 80 + "\n")
def mostrar_estatisticas_detalhadas(arvore):
    """
    Mostra estatísticas detalhadas sobre a árvore e suas correntes.
    """
    print("\n" + "=" * 80)
    print("ESTATISTICAS DETALHADAS")
    print("=" * 80)
    # Coleta todas as folhas
    folhas = []
    def coletar_folhas(node):
        if node.is_leaf():
            folhas.append(node.value)
        else:
            if node.left:
                coletar_folhas(node.left)
            if node.right:
                coletar_folhas(node.right)
    coletar_folhas(arvore)
    print(f"\nTotal de correntes filosoficas: {len(folhas)}")
    print("\nLista de todas as correntes identificaveis:")
    print("-" * 80)
    categorias = {
        "Racionalismo": [],
        "Empirismo": [],
        "Pragmatismo": [],
        "Hedonismo": [],
        "Existencialismo": [],
        "Niilismo": [],
        "Outras": []
    }
    for i, folha in enumerate(folhas, 1):
        nome = folha.split('\n')[0]
        print(f"{i:2d}. {nome}")   
        # Categoriza
        if "RACIONALISMO" in nome:
            categorias["Racionalismo"].append(nome)
        elif "EMPIRISMO" in nome or "POSITIVISMO" in nome:
            categorias["Empirismo"].append(nome)
        elif "PRAGMATISMO" in nome or "CONVENCIONALISMO" in nome:
            categorias["Pragmatismo"].append(nome)
        elif "HEDONISMO" in nome or "EPICURISMO" in nome or "UTILITARISMO" in nome:
            categorias["Hedonismo"].append(nome)
        elif "EXISTENCIALISMO" in nome:
            categorias["Existencialismo"].append(nome)
        elif "NIILISMO" in nome:
            categorias["Niilismo"].append(nome)
        else:
            categorias["Outras"].append(nome)
    print("\n" + "-" * 80)
    print("DISTRIBUICAO POR CATEGORIA:")
    print("-" * 80)
    for cat, items in categorias.items():
        if items:
            print(f"{cat}: {len(items)} correntes")
    
    print("=" * 80 + "\n")


def menu_principal():
    """
    Menu interativo para o usuário escolher o que fazer.
    """
    arvore = criar_arvore_filosofia_2() # Ou criar_arvore_filosofia()
    while True:
        print("\n" + "=" * 80)
        print("ARVORE DE DECISAO FILOSOFICA COMPLEXA - MENU PRINCIPAL")
        print("=" * 80)
        print("\n1. Fazer o questionario interativo (6 niveis de perguntas)")
        print("2. Visualizar a arvore completa graficamente")
        print("3. Ver informacoes sobre a estrutura da arvore")
        print("4. Ver estatisticas detalhadas das correntes")
        print("5. Sair")
        escolha = input("\n>>> Escolha uma opcao (1-5): ").strip()
        if escolha == '1':
            fazer_questionario_interativo(arvore)
            input("\nPressione ENTER para continuar...")
        elif escolha == '2':
            mostrar_arvore_completa(arvore)
            input("\nPressione ENTER para continuar...")
        elif escolha == '3':
            mostrar_info_arvore(arvore)
            input("\nPressione ENTER para continuar...")
        elif escolha == '4':
            mostrar_estatisticas_detalhadas(arvore)
            input("\nPressione ENTER para continuar...")
        elif escolha == '5':
            print("\nObrigado por usar o sistema! Ate logo!\n")
            break
        else:
            print("\nOpcao invalida! Tente novamente.")

if __name__ == "__main__":
    menu_principal()
