# ============================================================
# TEOREMA DO LIMITE CENTRAL: Distribuicao Amostral da Media
# Exemplo: populacao de salarios (assimetrica a direita)
# ============================================================


# ------------------------------------------------------------
# PASSO 1: Criando a populacao de salarios
# ------------------------------------------------------------

set.seed(42)

populacao <- rgamma(100000, shape = 2, scale = 1500) + 1000  # populacao com 100 mil salarios, assimetrica a direita

hist(populacao,
     breaks = 80,
     main = "Distribuicao da Populacao de Salarios",
     xlab = "Salario (R$)",
     col = "orange",
     border = "white")                                        # a distribuicao e claramente assimetrica a direita


# ------------------------------------------------------------
# PASSO 2: NA MAO, sorteando 3 amostras e calculando a media
# ------------------------------------------------------------

amostra_1 <- sample(populacao, size = 30)  # sorteia 30 pessoas da populacao
mean(amostra_1)                            # calcula a media amostral da amostra 1

amostra_2 <- sample(populacao, size = 30)
mean(amostra_2)                            # calcula a media amostral da amostra 2

amostra_3 <- sample(populacao, size = 30)
mean(amostra_3)                            # calcula a media amostral da amostra 3

# perceba que cada amostra da uma media diferente
# isso e a variabilidade amostral


# ------------------------------------------------------------
# PASSO 3: COM FOR, repetindo o experimento 10.000 vezes
# ------------------------------------------------------------

n_simulacoes <- 10000  # numero de vezes que vamos repetir o experimento
tamanho_amostra <- 30  # tamanho de cada amostra (n = 30 pessoas)

medias <- numeric(n_simulacoes)  # vetor vazio para guardar cada media calculada

for (i in 1:n_simulacoes) {
  amostra_i     <- sample(populacao, size = tamanho_amostra)  # sorteia uma amostra de 30 pessoas
  medias[i]     <- mean(amostra_i)                            # calcula e armazena a media dessa amostra
}

# medias agora contem 10.000 medias amostrais: a distribuicao amostral de X-barra


# ------------------------------------------------------------
# PASSO 4: Visualizando a distribuicao amostral de X-barra
# ------------------------------------------------------------

hist(medias,
     breaks = 60,
     main = "Distribuicao Amostral da Media (n = 30)",
     xlab = "Media Amostral", ylab = "Densidade",
     col = "steelblue",
     border = "white",
     probability = TRUE)                  # mesmo com a populacao assimetrica, X-barra tem formato de sino

curve(dnorm(x, mean = mean(medias), sd = sd(medias)),
      add = TRUE,
      col = "red",
      lwd = 2)                            # sobrepomos a curva normal: ela se encaixa muito bem


# ------------------------------------------------------------
# PASSO 5: Comparando populacao e distribuicao amostral
# ------------------------------------------------------------

par(mfrow = c(1, 2))                      # divide a janela em dois graficos lado a lado

hist(populacao,
     breaks = 80,
     main = "Populacao (assimetrica)",
     xlab = "Salario (R$)",
     col = "orange",
     border = "white",
     probability = TRUE)                  # populacao original: formato assimetrico

hist(medias,
     breaks = 60,
     main = "Distribuicao de X-barra (n = 30)",
     xlab = "Media Amostral",
     col = "steelblue",
     border = "white",
     probability = TRUE)

curve(dnorm(x, mean = mean(medias), sd = sd(medias)),
      add = TRUE, col = "red", lwd = 2)  # X-barra: formato normal, isso e o TLC em acao

par(mfrow = c(1, 1))                      # volta para janela unica


# ------------------------------------------------------------
# PASSO 6: Verificando as propriedades da media amostral
# ------------------------------------------------------------

mean(populacao)  # media real da populacao (mu)
mean(medias)     # media das 10.000 medias amostrais: deve ser muito proximo de mu (nao-viesada)

sd(populacao)                             # desvio padrao da populacao (sigma)
sd(medias)                                # desvio padrao de X-barra
sd(populacao) / sqrt(tamanho_amostra)     # valor teorico: sigma / raiz(n), deve bater com a linha acima (consistente)