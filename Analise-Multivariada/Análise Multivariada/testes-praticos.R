plot_vetores <- function(vetores, cores, labels, titulo, prod_interno) {
  df <- data.frame(
    x    = rep(0, nrow(vetores)),
    y    = rep(0, nrow(vetores)),
    xend = vetores[, 1],
    yend = vetores[, 2],
    cor  = cores,
    lab  = labels
  )
  
  lim <- 1.5
  
  ggplot(df) +
    geom_segment(
      aes(x = x, y = y, xend = xend, yend = yend, color = cor),
      arrow = arrow(length = unit(0.3, "cm")), linewidth = 1.2
    ) +
    geom_text(
      aes(x = xend * 1.25, y = yend * 1.25, label = lab, color = cor),
      size = 4.5, fontface = "bold"
    ) +
    geom_hline(yintercept = 0, color = "grey70") +
    geom_vline(xintercept = 0, color = "grey70") +
    scale_color_identity() +
    coord_fixed(xlim = c(-lim, lim), ylim = c(-lim, lim)) +
    labs(
      title    = titulo,
      subtitle = paste0("<x, y> = ", prod_interno),
      x = "x", y = "y"
    ) +
    theme_minimal(base_size = 12)
}

### ALGEBRA LINEAR

# Parte 1
x = c(3, 0, 0)
y = c(0, 2, 0)

prod_interno <- sum(x * y)
prod_interno2 <- as.numeric(x %*% y)

# Parte 2
A <- matrix(c(2, 1 ,-1,
              -3, -1, 2,
              -2, 1, 2), nrow=3, byrow = TRUE)

A

B <- c(8, -11, -3)

solucao <- solve(A, B)
solucao

# Parte 3
x <- c(1, 2, 3)
y <- c(4, 0, 1)

norma_x <- sqrt(sum(x^2))
norma_x2 <- norm(x, type = "2")

# Parte 4
x <- c(1, 2, 3)
x

plot(x)

# Parte 5

library(ggplot2)
library(patchwork)

# Caso 1: mesma direção
x1 <- c(1, 0)
y1 <- c(1, 0)
sum(x1 * y1)

v1 <- rbind(c(1, 0), c(1, 0))
p1 <- plot_vetores(v1,
                   cores = c("tomato", "steelblue"),
                   labels = c("x", "y"),
                   titulo = "Mesma direção  (Paralelos)",
                   prod_interno = 1)

# Caso 2: perpendiculares (ortogonais)
x2 <- c(1, 0)
y2 <- c(0, 1)
sum(x2 * y2)

v2 <- rbind(c(1, 0), c(0, 1))
p2 <- plot_vetores(v2,
                   cores = c("tomato", "steelblue"),
                   labels = c("x", "y"),
                   titulo = "Perpendiculares (Ortogonais)",
                   prod_interno = 0)

# Caso 3: direções opostas
x3 <- c(1, 0)
y3 <- c(-1, 0)
sum(x3 * y3)

v3 <- rbind(c(1, 0), c(-1, 0))
p3 <- plot_vetores(v3,
                   cores = c("tomato", "steelblue"),
                   labels = c("x", "y"),
                   titulo = "Direções opostas",
                   prod_interno = -1)

p1 | p2 | p3

# Parte 6
x <- c(2, 4, 4, 4, 5)
y <- c(1, 3, 4, 5, 7)

# correlação tradicional
cor(x, y)

# produto interno normalizado
sum(x * y) / (norm(x, "2") * norm(y, "2"))

# Parte 7

# duas variáveis perfeitamente correlacionadas
x <- c(1, 2, 3, 4, 5)
y <- x * 2  # y é exatamente o dobro de x — mesma direção

M <- cov(cbind(x, y))
cat("Matriz de covariância:\n"); print(M)
cat("\nDeterminante:", det(M), "\n")

# tenta inverter
solve(M)

# Parte 7.1

x <- c(1, 2, 3, 4, 5)
y <- x * 2   # mesma direção

M_mesmo <- cov(cbind(x, y))
cat("Mesma direção — det:", det(M_mesmo), "\n")

# e se y for independente?
z <- c(2, 1, 4, 3, 5)

M_indep <- cov(cbind(x, z))
cat("Independente — det:", det(M_indep), "\n")

# Parte 8

x <- c(1, 2, 3)
y <- c(4, 0, -1)

cos_theta   <- sum(x * y) / (norm(x, "2") * norm(y, "2"))
theta_rad   <- acos(cos_theta)
theta_graus <- theta_rad * 180 / pi

cat("cos(θ) =", round(cos_theta, 4), "\n")
cat("θ =", round(theta_graus, 2), "graus\n")

# Parte 9

a2d <- c(3, 1)
b2d <- c(1, 3)

cos_ab <- sum(a2d * b2d) / (norm(a2d, "2") * norm(b2d, "2"))
cat("Ângulo entre a e b:", round(acos(cos_ab) * 180/pi, 2), "graus\n")

vets <- rbind(a2d, b2d)

prod_int <- sum(a2d * b2d)

plot_vetores(vets,
             cores  = c("tomato", "steelblue"),
             labels = c("a = (3,1)", "b = (1,3)"),
             titulo = "Ângulo entre a e b — 53,13°",
             prod_interno = prod_int)

# Parte 10

x <- c(3, 4)
y <- c(1, 0)  # eixo x — a "parede"

proj    <- (sum(x * y) / sum(y * y)) * y
residuo <- x - proj

cat("Projeção de x sobre y:", proj, "\n")
cat("Resíduo (x - proj):", residuo, "\n")
cat("<resíduo, y> =", sum(residuo * y))

df <- data.frame(
  x    = c(0, 0, 0, proj[1]),
  y    = c(0, 0, 0, proj[2]),
  xend = c(x[1], y[1]*4, proj[1], x[1]),
  yend = c(x[2], y[2]*4, proj[2], x[2]),
  nome = c("x = (3,4)", "y = (1,0)", "proj = (3,0)", "resíduo = (0,4)"),
  cor  = c("tomato", "steelblue", "darkgreen", "purple")
)

ggplot(df) +
  geom_segment(
    aes(x = x, y = y, xend = xend, yend = yend, color = cor),
    arrow = arrow(length = unit(0.3, "cm")), linewidth = 1.2
  ) +
  geom_text(
    aes(x = xend * 1.1, y = yend * 1.1, label = nome, color = cor),
    size = 4, fontface = "bold"
  ) +
  scale_color_identity() +
  coord_fixed(xlim = c(-1, 5), ylim = c(-1, 5)) +
  geom_hline(yintercept = 0, color = "grey70") +
  geom_vline(xintercept = 0, color = "grey70") +
  labs(title = "Projeção ortogonal de x sobre y") +
  theme_minimal(base_size = 13)
