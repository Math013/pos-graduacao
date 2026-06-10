**DIRETRIZES DA ENTREGA FINAL**

Projeto Prático de Machine Learning de Ponta a Ponta

**1\. Visão Geral do Projeto**

A entrega final desta disciplina consiste no desenvolvimento de um projeto prático de Machine Learning de ponta a ponta (end-to-end). O objetivo é aplicar com rigor técnico as melhores práticas de mercado, metodologias e algoritmos discutidos e estruturados ao longo de nossas aulas.

Cada aluno deverá escolher um problema real com um conjunto de dados real.

**Entregar até dia 07 de junho.**

**2\. Formato e Componentes da Entrega**

A entrega é composta obrigatoriamente por dois arquivos complementares. A ausência de qualquer um dos componentes resultará na desqualificação do projeto.

**Componente A: Jupyter Notebook (Google Colab)**

O código completo do pipeline deve ser entregue via link ou arquivo descritivo do Google Colab. Este arquivo não deve ser apenas um amontoado de código, mas sim a memória de cálculo, estratégia e pensamento crítico do grupo. Ele deve conter blocos claros de Markdown explicando as hipóteses levantadas, os motivos das decisões técnicas e a interpretação dos outputs.

O código deve implementar obrigatoriamente as seguintes etapas:

* **Análise Exploratória de Dados (EDA):**  Verificação de distribuições, detecção analítica/visual de outliers, matriz de correlação e diagnóstico de dados ausentes.  
* **Feature Engineering:**  Imputação de nulos, tratamento de outliers, encodings adequados para alta e baixa cardinalidade (Target Encoding, One-Hot, etc.), criação de novas features de negócio e transformações de escala quando necessárias.  
* **Seleção de Features:**  Seleção rigorosa das variáveis preditoras utilizando estritamente o algoritmo Boruta (via BorutaPy ou BorutaShap) para enxugar o espaço dimensional.  
* **Modelagem Baseline:**  Implementação de um estimador simples (Linear, Logístico ou Dummy) para fixar a métrica de referência.  
* **Modelos Estado da Arte:**  Treinamento e avaliação comparativa de 4 modelos estado da arte baseados em árvores: Random Florest, XGBoost, LightGBM e CatBoost.  
* **Tunagem de Hiperparâmetros:**  Otimização fina dos parâmetros do melhor modelo inicial utilizando frameworks modernos (como Optuna) ou RandomizedSearchCV.  
* **Interpretabilidade Computacional (SHAP):**  Uso de técnicas avançadas de explicabilidade. Deve conter o gráfico global (SHAP Summary Plot) e a análise local de uma amostra individual de teste (SHAP Waterfall ou Force Plot) detalhando a força e a direção de cada feature para aquela predição específica.

**Componente B: Documentação Técnica (PDF)**

Um relatório técnico em formato PDF, redigido com linguagem formal e padrão executivo. Esta documentação traduz o esforço do notebook em um artefato corporativo de governança e estratégia de engenharia. A estrutura deve seguir estritamente as seções especificadas a seguir.

**3\. Estrutura Padrão da Documentação Técnica (PDF)**

O documento PDF deverá ser estruturado exatamente com as seguintes seções e conteúdos:

| 1\. Visão Geral do Projeto | Contextualização do problema, justificativa de negócio, objetivo técnico do algoritmo (tipo de problema) e as limitações de escopo estabelecidas. |
| :---- | :---- |
| **2\. Proposta de Implementação de Negócio** | Mapeamento dos KPIs de negócio gerados, detalhamento da ação operacional mapeada a partir do score, regras de decisão/filtros (guardrails) e estimativa de impacto financeiro (ROI). |
| **3\. Arquitetura de Dados e Engenharia** | Mapeamento das fontes de dados, definição da granularidade da linha, janelas temporais de observação/performance, e estratégias de Feature Engineering e tratamento adotadas. |
| **4\. Modelagem e Validação** | Justificativa da seleção por Boruta, registro do desempenho de todos os modelos testados (RF, XGBoost, LightGBM, CatBoost), estratégia de validação cruzada/Out-of-Time para mitigar overfitting, e definição do threshold de decisão ótimo. |
| **5\. Interpretabilidade e Governança** | Apresentação da Feature Importance e análise profunda dos impactos de SHAP (Global e Local) traduzidos para uma linguagem inteligível para stakeholders de negócio. |
| **6\. Formato de Deploy e MLOps** | Desenho da arquitetura de implantação: padrão de inferência (Batch ou API Real-time), estratégia de containerização (Docker), infraestrutura de nuvem utilizada e desenho do pipeline de monitoramento (Data Drift e Concept Drift). |

**4\. Critérios de Avaliação**

* **Rigor Metodológico:**  Serão avaliados o rigor estatístico na EDA, o tratamento correto de vazamento de dados (data leakage) no pré-processamento e a correta aplicação do algoritmo Boruta.  
* **Qualidade da Modelagem:**  A substituição de modelos lineares simples por ensembles de gradiente robustos (RF, XGBoost, LightGBM, CatBoost) e a eficiência do processo de tunagem fina de hiperparâmetros.  
* **Capacidade de Explicabilidade:**  Capacidade de traduzir os valores SHAP matemáticos em relatórios compreensíveis de transparência e governança de modelos.  
* **Visão de Engenharia e MLOps:**  Maturidade ao propor um modelo de deploy condizente com o cenário, prevendo custos de infraestrutura e alertas automatizados de degradação do modelo.  
* **Conexão com o Negócio:**  Alinhamento entre a performance matemática (Métricas técnicas) e o ganho real de eficiência ou receita esperada na operação da empresa.