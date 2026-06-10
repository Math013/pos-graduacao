# ============================================================
# Modelos Multivariados
# Analise de Componentes Principais
# Base mtcars
# Prof. Vinicius Osterne
# ============================================================

# install.packages(c("ggplot2", "reshape2", "ggrepel"))
library(ggplot2)
library(reshape2)
library(ggrepel)


# ------------------------------------------------------------
# Base de dados
# ------------------------------------------------------------

# A base mtcars ja vem no R.
# Ela contem informacoes de 32 modelos de carros.
#
# Variaveis:
# mpg  -> consumo em milhas por galao
# cyl  -> numero de cilindros
# disp -> cilindrada
# hp   -> potencia
# drat -> razao do eixo traseiro
# wt   -> peso
# qsec -> tempo para percorrer 1/4 de milha
# vs   -> tipo de motor
# am   -> tipo de transmissao
# gear -> numero de marchas
# carb -> numero de carburadores

dados <- mtcars

head(dados)
dim(dados)
summary(dados)

# ------------------------------------------------------------
# Padronizacao dos dados
# ------------------------------------------------------------

# A PCA e sensivel a escala.
# Como as variaveis estao em unidades diferentes, precisamos padronizar.

dados_pad <- scale(dados)

round(colMeans(dados_pad), 3)
round(apply(dados_pad, 2, sd), 3)

# ------------------------------------------------------------
# Ajuste da PCA
# ------------------------------------------------------------

pca <- prcomp(dados_pad, center = FALSE, scale. = FALSE)
summary(pca)

# ------------------------------------------------------------
# Autovalores e variancia explicada
# ------------------------------------------------------------

autovalores <- pca$sdev^2
prop_var <- autovalores / sum(autovalores)
prop_acum <- cumsum(prop_var)

tabela_pca <- data.frame(
  Componente = paste0("CP", 1:length(autovalores)),
  Autovalor = autovalores,
  Proporcao = prop_var,
  Acumulada = prop_acum
)

round(tabela_pca, 3)

# ------------------------------------------------------------
# Cargas dos componentes principais
# ------------------------------------------------------------

# As cargas mostram quais variaveis mais contribuem para cada componente.

round(pca$rotation, 3)

# ------------------------------------------------------------
# Scores dos carros
# ------------------------------------------------------------

# Os scores mostram a posicao de cada carro nos componentes principais.

scores <- as.data.frame(pca$x)
scores$Carro <- rownames(mtcars)

head(scores)

# ------------------------------------------------------------
# Scree plot
# ------------------------------------------------------------

df_scree <- data.frame(
  Componente = factor(paste0("CP", 1:length(autovalores)),
                      levels = paste0("CP", 1:length(autovalores))),
  Autovalor = autovalores
)

ggplot(df_scree, aes(x = Componente, y = Autovalor)) +
  geom_col(fill = "#E87722") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray40") +
  labs(
    title = "Scree Plot",
    subtitle = "Linha tracejada: criterio de Kaiser",
    x = "Componente principal",
    y = "Autovalor"
  ) +
  theme_minimal(base_size = 12)

# ------------------------------------------------------------
# Variancia explicada acumulada
# ------------------------------------------------------------

df_acum <- data.frame(
  Componente = 1:length(prop_acum),
  Acumulada = prop_acum
)

ggplot(df_acum, aes(x = Componente, y = Acumulada)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 3) +
  scale_x_continuous(breaks = 1:length(prop_acum)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    title = "Variancia explicada acumulada",
    x = "Numero de componentes",
    y = "Proporcao acumulada"
  ) +
  theme_minimal(base_size = 12)

# ------------------------------------------------------------
# Grafico dos carros nos dois primeiros componentes
# ------------------------------------------------------------

ggplot(scores, aes(x = PC1, y = PC2, label = Carro)) +
  geom_point(color = "#E87722", size = 2) +
  geom_text_repel(size = 3) +
  geom_hline(yintercept = 0, color = "gray60") +
  geom_vline(xintercept = 0, color = "gray60") +
  labs(
    title = "Carros nos dois primeiros componentes principais",
    x = "CP1",
    y = "CP2"
  ) +
  theme_minimal(base_size = 12)

# ------------------------------------------------------------
# Heatmap das cargas
# ------------------------------------------------------------

cargas <- as.data.frame(round(pca$rotation, 3))
cargas$Variavel <- rownames(cargas)

cargas_long <- melt(
  cargas,
  id.vars = "Variavel",
  variable.name = "Componente",
  value.name = "Carga"
)

ggplot(cargas_long, aes(x = Componente, y = Variavel, fill = Carga)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(Carga, 2)), size = 3.2) +
  scale_fill_gradient2(
    low = "steelblue",
    mid = "white",
    high = "#E87722",
    midpoint = 0,
    limits = c(-1, 1)
  ) +
  labs(
    title = "Cargas dos componentes principais",
    x = NULL,
    y = NULL,
    fill = "Carga"
  ) +
  theme_minimal(base_size = 12)

# ------------------------------------------------------------
# Interpretacao
# ------------------------------------------------------------

# CP1 geralmente separa carros mais potentes, pesados e com maior cilindrada
# de carros mais economicos e leves.
#
# CP2 pode capturar diferencas relacionadas a transmissao, marchas,
# tempo de aceleracao ou configuracao mecanica.
#
# A interpretacao deve focar nas variaveis com maiores cargas em modulo.
# O sinal pode ser invertido sem alterar o significado estatistico da PCA.