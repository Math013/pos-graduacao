# ============================================================
# Modelos Multivariados
# Analise Fatorial Exploratoria com Itens Likert
# Prof. Vinicius Osterne
# ============================================================

# ------------------------------------------------------------
# Pacotes utilizados
# ------------------------------------------------------------
library(ggplot2)
library(psych)
library(GPArotation)
library(gridExtra)
library(reshape2)

# ------------------------------------------------------------
# Questionario de qualidade de vida academica
# n = 400 estudantes universitarios, 12 itens
#
# Bloco A — Saude Fisica:
#   Q1: Pratico atividade fisica regularmente
#   Q2: Me sinto com energia para as atividades do dia
#   Q3: Durmo bem e acordo descansado
#   Q4: Minha alimentacao e equilibrada
#
# Bloco B — Saude Mental:
#   Q5: Me sinto ansioso com frequencia (atenção: item invertido)
#   Q6: Consigo me concentrar nas aulas
#   Q7: Me sinto motivado para estudar
#   Q8: Lido bem com a pressao das avaliacoes
#
# Bloco C — Vida Social:
#   Q9:  Tenho bons relacionamentos com colegas
#   Q10: Me sinto integrado a universidade
#   Q11: Participo de atividades extracurriculares
#   Q12: Tenho suporte social quando preciso
# ------------------------------------------------------------

# ------------------------------------------------------------
# Escala Likert utilizada
# ------------------------------------------------------------

# 1 = Discordo totalmente
# 2 = Discordo
# 3 = Nem concordo nem discordo
# 4 = Concordo
# 5 = Concordo totalmente

# ------------------------------------------------------------
# Simulacao dos dados
# ------------------------------------------------------------
set.seed(42)
n <- 400
F_fisica <- rnorm(n)
F_mental <- rnorm(n)
F_social <- rnorm(n)
Q1_cont  <-  0.80 * F_fisica + 0.60 * rnorm(n)
Q2_cont  <-  0.75 * F_fisica + 0.66 * rnorm(n)
Q3_cont  <-  0.70 * F_fisica + 0.71 * rnorm(n)
Q4_cont  <-  0.65 * F_fisica + 0.76 * rnorm(n)
Q5_cont  <- -0.72 * F_mental + 0.69 * rnorm(n)
Q6_cont  <-  0.78 * F_mental + 0.63 * rnorm(n)
Q7_cont  <-  0.75 * F_mental + 0.66 * rnorm(n)
Q8_cont  <-  0.68 * F_mental + 0.73 * rnorm(n)
Q9_cont  <-  0.76 * F_social + 0.65 * rnorm(n)
Q10_cont <-  0.80 * F_social + 0.60 * rnorm(n)
Q11_cont <-  0.60 * F_social + 0.80 * rnorm(n)
Q12_cont <-  0.72 * F_social + 0.69 * rnorm(n)
dados_cont <- data.frame(
  Q1_cont, Q2_cont, Q3_cont, Q4_cont,
  Q5_cont, Q6_cont, Q7_cont, Q8_cont,
  Q9_cont, Q10_cont, Q11_cont, Q12_cont
)
likertizar <- function(x) {
  as.numeric(cut(
    x,
    breaks = quantile(x, probs = seq(0, 1, length.out = 6)),
    labels = 1:5,
    include.lowest = TRUE
  ))
}
dados <- data.frame(lapply(dados_cont, likertizar))
colnames(dados) <- paste0("Q", 1:12)



# ------------------------------------------------------------
# Visao geral da base simulada
# ------------------------------------------------------------

# Primeiras linhas da base
head(dados, 10)

# Dimensao da base
# Esperamos 400 linhas e 12 colunas
dim(dados)

# Resumo das variaveis
summary(dados)

# Frequencia de respostas em cada item
lapply(dados, table)

# ------------------------------------------------------------
# Matriz de correlacoes policoricas
# ------------------------------------------------------------

# Como os itens sao ordinais em escala Likert, a correlacao de Pearson
# nao e a melhor opcao.
#
# A correlacao policorica assume que as respostas ordinais observadas
# vieram de variaveis continuas latentes.
#
# Por isso, ela e mais adequada para questionarios com itens Likert.

R_poly <- psych::polychoric(dados)$rho

# Visualizamos a matriz de correlacoes policoricas
round(R_poly, 2)

# Esperamos observar tres blocos de correlacao:
#
# Q1-Q4   -> Saude Fisica
# Q5-Q8   -> Saude Mental
# Q9-Q12  -> Vida Social
#
# Como Q5 e invertido, ele deve aparecer com correlacoes negativas
# em relacao a Q6, Q7 e Q8.

# ------------------------------------------------------------
# Correlacoes medias dentro de cada bloco teorico
# ------------------------------------------------------------

blocos <- list(
  Fisica = paste0("Q", 1:4),
  Mental = paste0("Q", 5:8),
  Social = paste0("Q", 9:12)
)

# Para cada bloco, calculamos a media das correlacoes internas.
# Isso ajuda a verificar se os itens de um mesmo bloco estao associados.

sapply(blocos, function(b) {
  r <- R_poly[b, b]
  round(mean(r[lower.tri(r)]), 3)
})

# ------------------------------------------------------------
# Diagnostico pre-analise fatorial
# ------------------------------------------------------------

# Antes de ajustar uma AFE, precisamos verificar se a matriz de correlacoes
# e adequada para fatoracao.

# ------------------------------------------------------------
# Teste de esfericidade de Bartlett
# ------------------------------------------------------------

# H0: a matriz de correlacoes e aproximadamente uma matriz identidade.
# Isso significaria que os itens nao se correlacionam entre si.
#
# Queremos rejeitar H0.
# Portanto, p < 0.05 indica que ha correlacoes suficientes para aplicar AFE.

cortest.bartlett(R_poly, n = nrow(dados))

# ------------------------------------------------------------
# Medida KMO
# ------------------------------------------------------------

# O KMO avalia a adequacao da amostra para analise fatorial.
# 
# Regra pratica:
# < 0.50  -> inadequado
# 0.50-0.59 -> ruim
# 0.60-0.69 -> mediocre
# 0.70-0.79 -> bom
# 0.80-0.89 -> otimo
# >= 0.90 -> excelente
#
# A análise fatorial funciona bem quando as variáveis
# compartilham variância comum

KMO(R_poly)

# KMO individual por item.
# Itens com MSA < 0.50 podem ser problematicos.

round(KMO(R_poly)$MSAi, 3)

# ------------------------------------------------------------
# Escolha do numero de fatores
# ------------------------------------------------------------

# Calculamos os autovalores da matriz de correlacoes policoricas.
# Cada autovalor indica quanta variancia e explicada por uma dimensao.

valores_proprios <- eigen(R_poly)$values
round(valores_proprios, 3)
# Quanta variância total dos dados é explicada por uma determinada
# dimensão latente (soma da 12)


# ------------------------------------------------------------
# Criterio de Kaiser
# ------------------------------------------------------------

# O criterio de Kaiser sugere reter fatores com autovalor maior que 1.
sum(valores_proprios > 1)

# ------------------------------------------------------------
# Variancia explicada acumulada
# ------------------------------------------------------------

# Como temos 12 itens, a soma dos autovalores e igual a 12.
# Aqui calculamos a proporcao acumulada da variancia explicada.

variancia_acum <- cumsum(valores_proprios / sum(valores_proprios))
round(variancia_acum, 3)

# ------------------------------------------------------------
# Analise paralela
# ------------------------------------------------------------

# A analise paralela compara os autovalores reais com autovalores
# obtidos de dados aleatorios.
#
# Retemos os fatores cujos autovalores reais superam os aleatorios.

fa.parallel(
  R_poly,
  n.obs = nrow(dados),
  fa = "fa",
  fm = "minres",
  main = "Analise Paralela"
)

# Esperamos reter 3 fatores, pois a base foi simulada com:
# Saude Fisica, Saude Mental e Vida Social.

# ------------------------------------------------------------
# Ajuste inicial sem rotacao
# ------------------------------------------------------------

# Primeiro ajustamos o modelo sem rotacao.
# Essa etapa e didatica: serve para mostrar que a solucao inicial
# pode ser dificil de interpretar.

fa_sem_rot <- fa(
  R_poly,
  nfactors = 3,
  rotate = "none",
  fm = "minres",
  n.obs = nrow(dados)
)

# Cargas fatoriais sem rotacao
round(fa_sem_rot$loadings, 3)

# ------------------------------------------------------------
# Ajuste com rotacao Varimax
# ------------------------------------------------------------

# A rotacao Varimax e ortogonal.
# Isso significa que os fatores sao assumidos como nao correlacionados.
#
# Ela busca uma estrutura simples:
# cada item deve carregar fortemente em um fator e fracamente nos demais.

fa_varimax <- fa(
  R_poly,
  nfactors = 3,
  rotate = "varimax",
  fm = "minres",
  n.obs = nrow(dados)
)

# Exibimos apenas cargas com modulo acima de 0.30.
# Isso facilita a leitura.

print(fa_varimax$loadings, cutoff = 0.30)

# ------------------------------------------------------------
# Comunalidades
# ------------------------------------------------------------

# A comunalidade indica a proporcao da variancia de cada item
# explicada pelos fatores comuns.
#
# Valores baixos, por exemplo abaixo de 0.40, podem indicar que o item
# nao esta sendo bem explicado pela estrutura fatorial.

round(fa_varimax$communality, 3)

# ------------------------------------------------------------
# Variancia explicada
# ------------------------------------------------------------

# Vaccounted mostra a variancia explicada por cada fator e a acumulada.

fa_varimax$Vaccounted

# ------------------------------------------------------------
# Ajuste com rotacao Oblimin
# ------------------------------------------------------------

# A rotacao Oblimin e obliqua.
# Isso significa que permite correlacao entre os fatores.
#
# Em ciencias humanas, sociais e saude, muitas vezes essa opcao
# e mais realista, pois dimensoes como saude fisica, mental e social
# podem estar correlacionadas.

fa_oblimin <- fa(
  R_poly,
  nfactors = 3,
  rotate = "oblimin",
  fm = "minres",
  n.obs = nrow(dados)
)

# Matriz de cargas apos rotacao obliqua
print(fa_oblimin$loadings, cutoff = 0.30)

# Matriz de correlacao entre os fatores
round(fa_oblimin$Phi, 3)

# Se as correlacoes entre fatores forem muito proximas de zero,
# a solucao Varimax pode ser suficiente.
#
# Se forem moderadas ou altas, a solucao Oblimin pode ser preferida.

# ------------------------------------------------------------
# Escores fatoriais
# ------------------------------------------------------------

# Os escores fatoriais estimam, para cada estudante, sua posicao
# em cada fator latente.
#
# Como usamos uma matriz de correlacao policorica no ajuste,
# vamos calcular os escores com factor.scores.

escores <- factor.scores(
  x = dados,
  f = fa_varimax,
  method = "regression"
)$scores

# Dimensao esperada:
# 400 estudantes x 3 fatores

dim(escores)

# Primeiros escores estimados
round(head(escores, 10), 3)

# Como a rotacao Varimax e ortogonal, esperamos fatores aproximadamente
# nao correlacionados.

round(cor(escores), 3)

# ------------------------------------------------------------
# Nomeando os fatores
# ------------------------------------------------------------

# Para nomear os fatores, olhamos quais itens apresentam maiores cargas
# em cada fator.
#
# Essa etapa e interpretativa e depende do conteudo dos itens.

cargas_matrix <- as.matrix(fa_varimax$loadings)

for (f in 1:3) {
  cat("\n--- Fator", f, "---\n")
  ord <- order(abs(cargas_matrix[, f]), decreasing = TRUE)
  print(round(cargas_matrix[ord, f], 3))
}

# Interpretacao esperada:
#
# Um fator deve agrupar Q1-Q4:
# Saude e Condicionamento Fisico
#
# Um fator deve agrupar Q5-Q8:
# Bem-Estar e Saude Mental
#
# Um fator deve agrupar Q9-Q12:
# Integracao e Vida Social
#
# Q5 deve aparecer com sinal contrario aos demais itens do bloco mental,
# pois foi simulado como item invertido.

# ------------------------------------------------------------
# Visualizacao: scree plot
# ------------------------------------------------------------

df_ev <- data.frame(
  fator = 1:length(valores_proprios),
  autovalor = valores_proprios
)

ggplot(df_ev, aes(x = fator, y = autovalor)) +
  geom_line(color = "gray50", linewidth = 0.8) +
  geom_point(aes(color = autovalor > 1), size = 3) +
  geom_hline(
    yintercept = 1,
    linetype = "dashed",
    color = "#E87722",
    linewidth = 0.8
  ) +
  scale_color_manual(
    values = c("gray40", "#E87722"),
    labels = c("Autovalor <= 1", "Autovalor > 1")
  ) +
  scale_x_continuous(breaks = 1:length(valores_proprios)) +
  labs(
    title = "Scree Plot — Qualidade de Vida Academica",
    subtitle = "Linha laranja: criterio de Kaiser",
    x = "Numero do fator",
    y = "Autovalor",
    color = NULL
  ) +
  theme_minimal(base_size = 12)

# ------------------------------------------------------------
# Visualizacao: heatmap das cargas fatoriais
# ------------------------------------------------------------

cargas_df <- as.data.frame(round(cargas_matrix, 3))
colnames(cargas_df) <- c("Fator 1", "Fator 2", "Fator 3")
cargas_df$item <- rownames(cargas_df)

cargas_long <- melt(
  cargas_df,
  id.vars = "item",
  variable.name = "fator",
  value.name = "carga"
)

ggplot(cargas_long, aes(x = fator, y = item, fill = carga)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(
    aes(label = ifelse(abs(carga) >= 0.30, round(carga, 2), "")),
    color = "black",
    size = 3.5
  ) +
  scale_fill_gradient2(
    low = "steelblue",
    mid = "white",
    high = "#E87722",
    midpoint = 0,
    limits = c(-1, 1)
  ) +
  labs(
    title = "Cargas fatoriais apos rotacao Varimax",
    subtitle = "Apenas cargas >= |0.30| exibidas nos tiles",
    x = NULL,
    y = NULL,
    fill = "Carga"
  ) +
  theme_minimal(base_size = 11)

# ------------------------------------------------------------
# Visualizacao: comunalidades por item
# ------------------------------------------------------------

df_com <- data.frame(
  item = names(fa_varimax$communality),
  comunalidade = fa_varimax$communality
)

df_com$bloco <- ifelse(
  df_com$item %in% paste0("Q", 1:4),
  "Saude Fisica",
  ifelse(
    df_com$item %in% paste0("Q", 5:8),
    "Saude Mental",
    "Vida Social"
  )
)

df_com <- df_com[order(df_com$comunalidade), ]
df_com$item <- factor(df_com$item, levels = df_com$item)

cores_bloco <- c(
  "Saude Fisica" = "#E87722",
  "Saude Mental" = "#f0a050",
  "Vida Social" = "gray55"
)

ggplot(df_com, aes(x = comunalidade, y = item, fill = bloco)) +
  geom_col(width = 0.7) +
  geom_vline(
    xintercept = 0.40,
    linetype = "dashed",
    color = "gray60",
    linewidth = 0.8
  ) +
  geom_vline(
    xintercept = 0.60,
    linetype = "dashed",
    color = "#E87722",
    linewidth = 0.8
  ) +
  scale_fill_manual(values = cores_bloco) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(
    title = "Comunalidades por item",
    subtitle = "Linha cinza: corte 0.40 | Linha laranja: corte 0.60",
    x = "Comunalidade",
    y = NULL,
    fill = "Bloco"
  ) +
  theme_minimal(base_size = 11)

# ------------------------------------------------------------
# Visualizacao: escores fatoriais por estudante
# ------------------------------------------------------------

df_scores <- as.data.frame(escores)
colnames(df_scores) <- c("Fator.1", "Fator.2", "Fator.3")

p1 <- ggplot(df_scores, aes(x = Fator.1, y = Fator.2)) +
  geom_point(color = "#E87722", alpha = 0.40, size = 1.8) +
  geom_hline(yintercept = 0, color = "gray50", linewidth = 0.5) +
  geom_vline(xintercept = 0, color = "gray50", linewidth = 0.5) +
  labs(
    title = "Fator 1 vs Fator 2",
    x = "Fator 1",
    y = "Fator 2"
  ) +
  theme_minimal(base_size = 11)

p2 <- ggplot(df_scores, aes(x = Fator.1, y = Fator.3)) +
  geom_point(color = "gray55", alpha = 0.40, size = 1.8) +
  geom_hline(yintercept = 0, color = "gray50", linewidth = 0.5) +
  geom_vline(xintercept = 0, color = "gray50", linewidth = 0.5) +
  labs(
    title = "Fator 1 vs Fator 3",
    x = "Fator 1",
    y = "Fator 3"
  ) +
  theme_minimal(base_size = 11)

grid.arrange(
  p1,
  p2,
  ncol = 2,
  top = grid::textGrob(
    "Escores fatoriais: cada ponto representa um estudante",
    gp = grid::gpar(fontface = "bold", fontsize = 12)
  )
)