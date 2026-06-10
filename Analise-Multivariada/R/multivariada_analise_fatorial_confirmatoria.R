# ============================================================
# Modelos Multivariados
# Analise Fatorial Confirmatoria com Itens Likert
# Prof. Vinicius Osterne
# ============================================================

# ------------------------------------------------------------
# Pacotes utilizados
# ------------------------------------------------------------
library(lavaan)
library(semPlot)
library(psych)
library(ggplot2)

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
#   Q5: Me sinto ansioso com frequencia (item invertido)
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

# Fatores latentes nao observaveis
F_fisica <- rnorm(n)
F_mental <- rnorm(n)
F_social <- rnorm(n)

# ------------------------------------------------------------
# Geracao dos itens continuos latentes
# ------------------------------------------------------------

# A ideia e que por tras de cada resposta Likert existe uma resposta
# continua latente. Depois transformamos essa resposta continua em
# categorias ordinais de 1 a 5.

# Bloco A: Saude Fisica
Q1_cont  <-  0.80 * F_fisica + 0.60 * rnorm(n)
Q2_cont  <-  0.75 * F_fisica + 0.66 * rnorm(n)
Q3_cont  <-  0.70 * F_fisica + 0.71 * rnorm(n)
Q4_cont  <-  0.65 * F_fisica + 0.76 * rnorm(n)

# Bloco B: Saude Mental
# Q5 e item invertido. Por isso aparece com sinal negativo.
Q5_cont  <- -0.72 * F_mental + 0.69 * rnorm(n)
Q6_cont  <-  0.78 * F_mental + 0.63 * rnorm(n)
Q7_cont  <-  0.75 * F_mental + 0.66 * rnorm(n)
Q8_cont  <-  0.68 * F_mental + 0.73 * rnorm(n)

# Bloco C: Vida Social
Q9_cont  <-  0.76 * F_social + 0.65 * rnorm(n)
Q10_cont <-  0.80 * F_social + 0.60 * rnorm(n)
Q11_cont <-  0.60 * F_social + 0.80 * rnorm(n)
Q12_cont <-  0.72 * F_social + 0.69 * rnorm(n)

dados_cont <- data.frame(
  Q1_cont, Q2_cont, Q3_cont, Q4_cont,
  Q5_cont, Q6_cont, Q7_cont, Q8_cont,
  Q9_cont, Q10_cont, Q11_cont, Q12_cont
)

# ------------------------------------------------------------
# Funcao para transformar variaveis continuas em Likert
# ------------------------------------------------------------

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
# Visao geral dos dados
# ------------------------------------------------------------

head(dados, 10)
dim(dados)
summary(dados)

# Frequencia das respostas por item
lapply(dados, table)

# ------------------------------------------------------------
# Tratamento do item invertido
# ------------------------------------------------------------

# Q5 foi formulado de modo negativo:
# "Me sinto ansioso com frequencia"
#
# Como os demais itens de Saude Mental estao no sentido positivo,
# vamos inverter Q5 para que todos os itens apontem na mesma direcao.
#
# Em escala de 1 a 5, a inversao e feita por:
#
# novo_item = 6 - item_antigo

dados$Q5_inv <- 6 - dados$Q5

# Vamos manter Q5 original apenas para referencia.
# Na AFC, usaremos Q5_inv no fator Saude Mental.

head(dados[, c("Q5", "Q5_inv")], 10)

# ------------------------------------------------------------
# Definicao dos itens ordinais
# ------------------------------------------------------------

# Como os itens sao Likert, informamos ao lavaan que eles sao ordinais.
# Isso faz com que o modelo seja estimado com um estimador apropriado,
# geralmente WLSMV.

itens_ordinais <- c(
  "Q1", "Q2", "Q3", "Q4",
  "Q5_inv", "Q6", "Q7", "Q8",
  "Q9", "Q10", "Q11", "Q12"
)

# ------------------------------------------------------------
# Modelo confirmatorio teorico
# ------------------------------------------------------------

# Diferente da AFE, na AFC nos ja especificamos antes quais itens
# pertencem a quais fatores.
#
# Aqui estamos testando uma estrutura com tres fatores:
#
# 1. Saude_Fisica
# 2. Saude_Mental
# 3. Vida_Social

modelo_cfa <- '
  Saude_Fisica =~ Q1 + Q2 + Q3 + Q4

  Saude_Mental =~ Q5_inv + Q6 + Q7 + Q8

  Vida_Social =~ Q9 + Q10 + Q11 + Q12
'

# ------------------------------------------------------------
# Ajuste do modelo
# ------------------------------------------------------------

# ordered = itens_ordinais informa que os itens sao ordinais.
#
# estimator = "WLSMV" e recomendado para itens categoricos ordinais,
# como escalas Likert.
#
# std.lv = TRUE fixa a variancia dos fatores em 1.
# Isso facilita a interpretacao das cargas padronizadas.

fit_cfa <- cfa(
  model = modelo_cfa,
  data = dados,
  ordered = itens_ordinais,
  estimator = "WLSMV",
  std.lv = TRUE
)

# ------------------------------------------------------------
# Resumo do modelo
# ------------------------------------------------------------

# fit.measures = TRUE mostra os indices de ajuste.
# standardized = TRUE mostra as cargas padronizadas.
# rsquare = TRUE mostra o R2 de cada item.

summary(
  fit_cfa,
  fit.measures = TRUE,
  standardized = TRUE,
  rsquare = TRUE
)

# ------------------------------------------------------------
# Principais indices de ajuste
# ------------------------------------------------------------

# CFI e TLI:
# valores proximos de 0.90 indicam ajuste aceitavel
# valores proximos de 0.95 indicam bom ajuste
#
# RMSEA:
# valores abaixo de 0.08 indicam ajuste aceitavel
# valores abaixo de 0.06 indicam bom ajuste
#
# SRMR:
# valores abaixo de 0.08 costumam indicar bom ajuste

fitMeasures(
  fit_cfa,
  c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr")
)

# ------------------------------------------------------------
# Cargas fatoriais padronizadas
# ------------------------------------------------------------

# As cargas mostram o quanto cada item esta associado ao seu fator.
#
# Regra pratica:
# cargas abaixo de 0.30 sao fracas
# cargas entre 0.30 e 0.50 sao moderadas
# cargas acima de 0.50 sao boas
# cargas acima de 0.70 sao fortes

cargas_padronizadas <- standardizedSolution(fit_cfa)

cargas_padronizadas[
  cargas_padronizadas$op == "=~",
  c("lhs", "rhs", "est.std", "pvalue")
]

# ------------------------------------------------------------
# R2 dos itens
# ------------------------------------------------------------

# O R2 indica a proporcao da variancia do item explicada pelo fator.
#
# Por exemplo:
# R2 = 0.64 significa que 64% da variabilidade do item
# e explicada pelo fator latente correspondente.

inspect(fit_cfa, "r2")

# ------------------------------------------------------------
# Correlacoes entre fatores
# ------------------------------------------------------------

# Como estamos em uma AFC com tres fatores, tambem podemos avaliar
# se os fatores latentes se correlacionam entre si.

lavInspect(fit_cfa, "cor.lv")

# ------------------------------------------------------------
# Residuos do modelo
# ------------------------------------------------------------

# Os residuos mostram diferencas entre correlacoes observadas
# e correlacoes reproduzidas pelo modelo.
#
# Residuos altos podem indicar problemas locais de ajuste.

residuos <- residuals(fit_cfa, type = "cor")$cov

round(residuos, 3)

# ------------------------------------------------------------
# Indices de modificacao
# ------------------------------------------------------------

# Os indices de modificacao sugerem parametros que, se liberados,
# poderiam melhorar o ajuste do modelo.
#
# Cuidado:
# indices de modificacao nao devem ser usados mecanicamente.
# So devemos modificar o modelo se houver justificativa teorica.

modindices(fit_cfa, sort = TRUE, minimum.value = 10)

# ------------------------------------------------------------
# Diagrama do modelo
# ------------------------------------------------------------

# O grafico mostra os fatores latentes, os itens observados
# e as cargas fatoriais padronizadas.

semPaths(
  fit_cfa,
  what = "std",
  whatLabels = "std",
  layout = "tree",
  rotation = 2,
  sizeMan = 5,
  sizeLat = 7,
  edge.label.cex = 0.8,
  residuals = FALSE,
  intercepts = FALSE,
  thresholds = FALSE
)

# ------------------------------------------------------------
# Comparacao com modelo alternativo de um fator
# ------------------------------------------------------------

# Um teste didatico importante e comparar o modelo teorico de tres fatores
# com um modelo mais simples de apenas um fator geral.
#
# Se o modelo de tres fatores ajustar melhor, isso fortalece a ideia
# de que existem dimensoes distintas no questionario.

modelo_1fator <- '
  Qualidade_Vida =~ Q1 + Q2 + Q3 + Q4 +
                    Q5_inv + Q6 + Q7 + Q8 +
                    Q9 + Q10 + Q11 + Q12
'

fit_1fator <- cfa(
  model = modelo_1fator,
  data = dados,
  ordered = itens_ordinais,
  estimator = "WLSMV",
  std.lv = TRUE
)

# Indices de ajuste do modelo de um fator
fitMeasures(
  fit_1fator,
  c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr")
)

# Indices de ajuste do modelo de tres fatores
fitMeasures(
  fit_cfa,
  c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr")
)

# ------------------------------------------------------------
# Tabela comparativa dos dois modelos
# ------------------------------------------------------------

comparacao <- data.frame(
  Modelo = c("Um fator", "Tres fatores"),
  CFI = c(
    fitMeasures(fit_1fator, "cfi"),
    fitMeasures(fit_cfa, "cfi")
  ),
  TLI = c(
    fitMeasures(fit_1fator, "tli"),
    fitMeasures(fit_cfa, "tli")
  ),
  RMSEA = c(
    fitMeasures(fit_1fator, "rmsea"),
    fitMeasures(fit_cfa, "rmsea")
  ),
  SRMR = c(
    fitMeasures(fit_1fator, "srmr"),
    fitMeasures(fit_cfa, "srmr")
  )
)

comparacao

# ------------------------------------------------------------
# Visualizacao dos indices de ajuste
# ------------------------------------------------------------

comparacao_long <- reshape(
  comparacao,
  varying = c("CFI", "TLI", "RMSEA", "SRMR"),
  v.names = "Valor",
  timevar = "Indice",
  times = c("CFI", "TLI", "RMSEA", "SRMR"),
  direction = "long"
)

ggplot(comparacao_long, aes(x = Indice, y = Valor, fill = Modelo)) +
  geom_col(position = "dodge") +
  labs(
    title = "Comparacao dos indices de ajuste",
    subtitle = "Modelo de um fator versus modelo teorico de tres fatores",
    x = NULL,
    y = "Valor do indice",
    fill = "Modelo"
  ) +
  theme_minimal(base_size = 12)
