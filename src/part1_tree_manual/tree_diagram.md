flowchart TD
      start([Inicio])--> q1{Você acredita que existe uma verdade objetiva e universal?}
      
      q1 -- Sim --> q2{A razão é superior à experiência sensorial como fonte de conhecimento?}
      q1 -- Não --> q3{A existência humana possui algum valor ou significado inerente?}
      q2 -- Sim --> q4{O conhecimento verdadeiro pode ser alcançado pela intuição intelectual?}
      q2 -- Não --> q5{A verdade de uma ideia depende de suas consequências práticas?}
      q3 -- Sim --> q19{A felicidade e o bem-estar são objetivos legítimos da filosofia?}
      q3 -- Não --> q26{A liberdade individual é mais importante que o bem coletivo?}
      q4 -- Sim --> q6{As Formas ou Ideias platônicas existem independentemente da mente?}
      q4 -- Não --> q7{Todo conhecimento deriva da experiência, mesmo que processado pela razão?}
      q5 -- Sim --> q13{A utilidade é o único critério válido para avaliar teorias?}
      q5 -- Não --> q14{A linguagem e a análise lógica são centrais para a filosofia?}
      q6 -- Sim --> q8{O mundo sensível é apenas sombra de uma realidade superior?}
      q6 -- Não --> q9{A dúvida metódica é o caminho para a certeza?}
      q7 -- Sim --> q11{A mente já possui conhecimento inato ao nascer?}
      q7 -- Não --> q12{A percepção sensorial é a única fonte de ideias válidas?}
      q13 -- Sim --> q15{As teorias científicas são apenas instrumentos úteis, não verdades?}
      q13 -- Não --> q16{A verdade é relativa ao contexto e à comunidade?}
      q14 -- Sim --> q17{A verificação empírica é essencial para proposições significativas?}
      q14 -- Não --> q18{Os problemas filosóficos são pseudoproblemas linguísticos?} 
      q19 -- Sim --> q20{O prazer é o bem supremo e deve ser maximizado?}
      q19 -- Não --> q23{A virtude e o dever são mais importantes que a felicidade pessoal?}
      q20 -- Sim --> q21{O prazer imediato é preferível ao prazer calculado?}
      q20 -- Não --> q22{A ataraxia é alcançada pelo prazer moderado?}
      q23 -- Sim --> q24{Maximizar o prazer coletivo justifica sacrifícios individuais?}
      q23 -- Não --> q25{A virtude por si mesma traz felicidade verdadeira?}
      q26 -- Sim --> q27{A autenticidade e a escolha pessoal definem a existência?}
      q26 -- Não --> q30{Todos os valores e significados são construções arbitrárias?}
      q27 -- Sim --> q28{A angústia da liberdade é inseparável da condição humana?}
      q27 -- Não --> q29{O absurdo da existência deve ser reconhecido e abraçado?}
      q30 -- Sim --> q31{A morte de Deus liberta o indivíduo para criar valores próprios?}
      q30 -- Não --> q32{A busca por significado é fútil e deve ser abandonada?}

      q8 -- Sim --> sA[RACIONALISMO PLATÔNICO]
      q8 -- Não --> sB[RACIONALISMO CARTESIANO]  
      q9 -- Sim --> sC[RACIONALISMO SPINOZISTA]
      q9 -- Não --> sD[EMPIRISMO LOCKEANO]
      q11 -- Sim --> sE[PRAGMATISMO INSTRUMENTALISTA]
      q11 -- Não --> sF[PRAGMATISMO CLÁSSICO]
      q12 -- Sim --> sG[POSITIVISMO LÓGICO]
      q12 -- Não --> sH[NEOPRAGMATISMO]
      q15 -- Sim --> sI[HEDONISMO CIRENAICO]
      q15 -- Não --> sJ[EPICURISMO]
      q16 -- Sim --> SK[UTILITARISMO HEDONISTA]
      q16 -- Não --> sL[EUDEMONISMO]
      q17 -- Sim --> sM[EXISTENCIALISMO SARTREANO]
      q17 -- Não --> sN[EXISTENCIALISMO CAMUSIANO]
      q18 -- Sim --> sO[NIILISMO ATIVO]
      q18 -- Não --> sP[NIILISMO PASSIVO]
      q21 -- Sim --> sR[RACIONALISMO LEIBNIZIANO]
      q21 -- Não --> sS[EMPIRISMO HUMEANO]
      q22 -- Sim --> sT[EMPIRISMO BERKELEYANO]
      q22 -- Não --> sU[POSITIVISMO COMTEANO]
      q24 -- Sim --> sV[PRAGMATISMO RADICAL]
      q24 -- Não --> sW[CONVENCIONALISMO]
      q25 -- Sim --> sX[FILOSOFIA ANALITICA]
      q25 -- Não --> sY[FILOSOFIA DA LINGUAGEM ORDINARIA]
      q28 -- Sim --> sZ[HEDONISMO PSICOLÓGICO]
      q28 -- Não --> sAA[EPICURISMO MODERNO]
      q29 -- Sim --> sAB[CONSEQUENCIALISMO]
      q29 -- Não --> sAC[ARISTOTELISMO]
      q31 -- Sim --> sAD[EXISTENCIALISMO KIERKEGAARDIANO] 
      q31 -- Sim --> sAE[EXISTENCIALISMO HEIDEGGERIANO]
      q32 -- Sim --> sAF[NIILISMO MORAL]
      q32 -- Não --> sAG[NIILISMO EPISTEMOLOGICO]