# ============================================================
# Inferencia Multivariada — Metodo 3
# T2 de Hotelling: populacoes pareadas
# Exemplo REAL: medicoes cardiovasculares antes e apos exercicio
# Fonte: Johnson & Wichern (2007), Applied Multivariate
#        Statistical Analysis, 6a ed., Tabela 6.1
# Prof. Vinicius Osterne
# ============================================================

# install.packages(c("MASS", "ggplot2", "Hotelling", "tidyr", "dplyr"))

library(MASS)
library(ggplot2)
library(Hotelling)
library(tidyr)
library(dplyr)

# ------------------------------------------------------------
# Dados reais: 11 individuos
# Variaveis: pressao sistolica, diastolica e frequencia cardiaca
# Medicoes em repouso (antes) e apos exercicio padronizado (depois)
# ------------------------------------------------------------

antes <- matrix(c(
  130, 91, 70,
  142, 89, 76,
  138, 87, 72,
  145, 82, 79,
  162, 96, 85,
  142, 86, 76,
  170, 90, 90,
  124, 78, 73,
  158, 91, 82,
  154, 94, 85,
  162, 95, 90
), nrow = 11, byrow = TRUE)

depois <- matrix(c(
  138, 94,  80,
  152, 95,  90,
  145, 92,  82,
  155, 87,  88,
  168, 100, 95,
  148, 90,  84,
  180, 96, 100,
  130, 82,  82,
  165, 96,  92,
  162, 98,  94,
  170, 100, 100
), nrow = 11, byrow = TRUE)

colnames(antes)  <- c("sist", "diast", "freq_cardiaca")
colnames(depois) <- c("sist", "diast", "freq_cardiaca")

head(antes)
head(depois)
depois - antes
# ------------------------------------------------------------
# Calcular as diferencas (depois - antes)
# ------------------------------------------------------------

diferencas <- depois - antes
colnames(diferencas) <- c("delta_sist", "delta_diast", "delta_freq")

n_pares <- nrow(diferencas)
p3      <- ncol(diferencas)

# Estimativas sobre as diferencas
Dbarra <- colMeans(diferencas)
S_D    <- cov(diferencas)

round(Dbarra, 4)   # vetor de medias das diferencas
round(S_D, 4)      # matriz de covariancias das diferencas

# ------------------------------------------------------------
# Estatistica T2 (caso pareado = T2 de uma populacao com mu0 = 0)
# H0: mu_D = 0  (exercicio nao altera as medidas)
# H1: mu_D != 0 (exercicio altera ao menos uma medida)
# ------------------------------------------------------------

T2_par  <- as.numeric(n_pares * t(Dbarra) %*% solve(S_D) %*% Dbarra)
fator3  <- ((n_pares - 1) * p3) / (n_pares - p3)
F_obs3  <- T2_par / fator3
pvalor3 <- pf(F_obs3, df1 = p3, df2 = n_pares - p3, lower.tail = FALSE)

T2_par    # estatistica T2 de Hotelling
F_obs3    # estatistica F equivalente
pvalor3   # p-valor

# Valor critico a 5%
T2_crit3 <- fator3 * qf(0.05, df1 = p3, df2 = n_pares - p3,
                        lower.tail = FALSE)
T2_crit3  # rejeita H0 se T2_par > T2_crit3

# Confirmacao via pacote DescTools (aceita caso de uma amostra)
# install.packages("DescTools")
library(DescTools)
HotellingsT2Test(as.data.frame(diferencas), mu = rep(0, p3))

# ------------------------------------------------------------
# Visualizacao 1: perfis individuais antes e depois
# ------------------------------------------------------------

df_perfis <- data.frame(
  individuo = rep(1:n_pares, 2),
  momento   = rep(c("Antes", "Depois"), each = n_pares),
  sist      = c(antes[, 1], depois[, 1]),
  diast     = c(antes[, 2], depois[, 2]),
  freq      = c(antes[, 3], depois[, 3])
)

df_perfis_long <- df_perfis |>
  pivot_longer(cols = c(sist, diast, freq),
               names_to  = "variavel",
               values_to = "valor") |>
  mutate(momento = factor(momento, levels = c("Antes", "Depois")))

ggplot(df_perfis_long,
       aes(x = momento, y = valor, group = individuo)) +
  geom_line(color = "gray50", alpha = 0.5, linewidth = 0.5) +
  geom_point(aes(color = momento), size = 2, alpha = 0.8) +
  stat_summary(aes(group = 1), fun = mean,
               geom = "line", color = "orange", linewidth = 1.2) +
  stat_summary(aes(group = 1), fun = mean,
               geom = "point", color = "orange", size = 3.5) +
  scale_color_manual(values = c("Antes"  = "gray60",
                                "Depois" = "#E87722")) +
  facet_wrap(~variavel, scales = "free_y",
             labeller = labeller(variavel = c(
               diast = "Diastolica (mmHg)",
               freq  = "Freq. cardiaca (bpm)",
               sist  = "Sistolica (mmHg)"
             ))) +
  labs(
    title    = "Perfis individuais antes e depois do exercicio",
    subtitle = "Fonte: Johnson & Wichern (2007), Tab. 6.1 | Linha laranja = media",
    x = NULL, y = "Valor", color = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background  = element_rect(fill = "#1a1a1a", color = NA),
    panel.background = element_rect(fill = "#1a1a1a", color = NA),
    panel.grid       = element_line(color = "#2e2e2e"),
    text             = element_text(color = "gray80"),
    strip.text       = element_text(color = "gray90", face = "bold"),
    axis.text        = element_text(color = "gray60"),
    legend.position  = "none",
    plot.title       = element_text(color = "gray95", face = "bold"),
    plot.subtitle    = element_text(color = "gray60", size = 9)
  )

# ------------------------------------------------------------
# Visualizacao 2: distribuicao das diferencas com linha em zero
# ------------------------------------------------------------

df_dif_long <- as.data.frame(diferencas) |>
  pivot_longer(cols = everything(),
               names_to  = "variavel",
               values_to = "diferenca")

ggplot(df_dif_long, aes(x = diferenca)) +
  geom_histogram(aes(y = after_stat(density)),
                 bins = 8, fill = "#E87722", alpha = 0.6, color = NA) +
  geom_density(color = "orange", linewidth = 0.9) +
  geom_vline(xintercept = 0, color = "white",
             linetype = "dashed", linewidth = 0.8) +
  facet_wrap(~variavel, scales = "free",
             labeller = labeller(variavel = c(
               delta_diast = "Diastolica",
               delta_freq  = "Freq. cardiaca",
               delta_sist  = "Sistolica"
             ))) +
  labs(
    title    = "Distribuicao das diferencas (depois - antes)",
    subtitle = "Linha branca tracejada em zero: ausencia de efeito do exercicio",
    x = "Diferenca", y = "Densidade"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.background  = element_rect(fill = "#1a1a1a", color = NA),
    panel.background = element_rect(fill = "#1a1a1a", color = NA),
    panel.grid       = element_line(color = "#2e2e2e"),
    text             = element_text(color = "gray80"),
    strip.text       = element_text(color = "gray90", face = "bold"),
    axis.text        = element_text(color = "gray60"),
    plot.title       = element_text(color = "gray95", face = "bold"),
    plot.subtitle    = element_text(color = "gray60", size = 9)
  )