# =============================================================================
# Álgebra Linear
# Prof. Vinícius Osterne, PhD
# =============================================================================

# -----------------------------------------------------------------------------
# 1. ESCALAR
# -----------------------------------------------------------------------------

# Um escalar é simplesmente um número real — um único valor
a <- 5
b <- -3.14
c <- sqrt(2)



# -----------------------------------------------------------------------------
# 2. VETOR
# -----------------------------------------------------------------------------

# Um vetor é uma sequência ordenada de escalares
x <- c(1, 2, 3)
y <- c(4, 0, -1)

# Comprimento (norma euclidiana) de um vetor
norma_x <- sqrt(sum(x^2))

# Usando a função embutida
norma_x2 <- norm(x, type = "2")


# -----------------------------------------------------------------------------
# 3. PRODUTO INTERNO
# -----------------------------------------------------------------------------

prod_interno <- sum(x * y)

# Também pode ser feito com operador matricial:
prod_interno2 <- as.numeric(x %*% y)

# Exemplo 1: vetores ortogonais (produto interno = 0)
u <- c(1, 0)
v <- c(0, 1)
sum(u * v)

# Exemplo 2: vetores paralelos (produto interno = produto das normas)
u2 <- c(1, 2, 3)
v2 <- c(2, 4, 6)  # v2 = 2 * u2
sum(u2 * v2)
norm(u2, "2") * norm(v2, "2") #vetores paralelos


# Exemplo 3: interpretação estatística — produto interno e covariância
# Se x e y são vetores centrados, <x, y> / (n-1) é a covariância amostral
n   <- 5
x3  <- c(2, 4, 4, 4, 5) - mean(c(2, 4, 4, 4, 5))  # centralizado
y3  <- c(1, 3, 4, 5, 7) - mean(c(1, 3, 4, 5, 7))   # centralizado
cov_xy <- sum(x3 * y3) / (n - 1)
cov_xy #Covariância via produto interno
cov(c(2,4,4,4,5), c(1,3,4,5,7)) #Covariância via cov()


# -----------------------------------------------------------------------------
# 4. ÂNGULO ENTRE VETORES
# -----------------------------------------------------------------------------

# x <- c(1, 2, 3)
# y <- c(4, 0, -1)
x <- c(1, 0)
y <- c(1, 1)

cos_theta <- sum(x * y) / (norm(x, "2") * norm(y, "2"))
theta_rad <- acos(cos_theta)
theta_graus <- theta_rad * 180 / pi

round(cos_theta, 4) #cos(theta)
round(theta_graus, 2) #graus


# Visualização: ângulo entre dois vetores 2D
a2d <- c(3, 1)
b2d <- c(1, 3)
cos_ab <- sum(a2d * b2d) / (norm(a2d, "2") * norm(b2d, "2"))
round(acos(cos_ab) * 180/pi, 2) #Ângulo entre a e b


# -----------------------------------------------------------------------------
# 5. PROJEÇÃO ORTOGONAL
# -----------------------------------------------------------------------------

# A projeção de x sobre y é: proj_y(x) = (<x,y> / <y,y>) * y
x <- c(3, 4)
y <- c(1, 0)  # eixo x

proj <- (sum(x * y) / sum(y * y)) * y
proj

# Verificando ortogonalidade: (x - proj) deve ser perpendicular a y
residuo <- x - proj
residuo
sum(residuo * y)


# Exemplo com vetor genérico
x2 <- c(2, 3, 1)
y2 <- c(1, 1, 1)
proj2 <- (sum(x2 * y2) / sum(y2 * y2)) * y2
proj2



# -----------------------------------------------------------------------------
# 6. MATRIZES
# -----------------------------------------------------------------------------

# Uma matriz é um arranjo bidimensional de escalares
A <- matrix(c(1, 2, 3,
              4, 5, 6,
              7, 8, 9), nrow = 3, byrow = TRUE)
A

B <- matrix(c(9, 8, 7,
              6, 5, 4,
              3, 2, 1), nrow = 3, byrow = TRUE)

B



# -----------------------------------------------------------------------------
# 7. ADIÇÃO E MULTIPLICAÇÃO DE MATRIZES
# -----------------------------------------------------------------------------

# Adição (elemento a elemento — mesmas dimensões)
print(A + B)

# Multiplicação por escalar
print(2 * A)

# Multiplicação matricial: A %*% B  (não confundir com A * B)
print(A %*% B)


# Exemplo com dimensões compatíveis
C <- matrix(c(1, 2,
              3, 4,
              5, 6), nrow = 3, byrow = TRUE)  # 3x2

D <- matrix(c(1, 0,
              0, 1), nrow = 2, byrow = TRUE)  # 2x2

C
D
C %*% D


# -----------------------------------------------------------------------------
# 8. TRANSPOSTA
# -----------------------------------------------------------------------------

t(A)
t(C)

# Propriedade: (A %*% B)^T = B^T %*% A^T
lhs <- t(A %*% B)
rhs <- t(B) %*% t(A)
all.equal(lhs, rhs)


# -----------------------------------------------------------------------------
# 9. TRAÇO DE UMA MATRIZ
# -----------------------------------------------------------------------------

# Traço = soma dos elementos da diagonal principal
traco_A <- sum(diag(A))
traco_A

# Propriedade: tr(A + B) = tr(A) + tr(B)
cat("tr(A) + tr(B):", sum(diag(A)) + sum(diag(B)), "\n")
cat("tr(A + B):", sum(diag(A + B)), "\n")

# Propriedade: tr(A %*% B) = tr(B %*% A)
cat("tr(A %*% B):", sum(diag(A %*% B)), "\n")
cat("tr(B %*% A):", sum(diag(B %*% A)), "\n")


# -----------------------------------------------------------------------------
# 10. DETERMINANTE
# -----------------------------------------------------------------------------

M <- matrix(c(2, 1,
              5, 3), nrow = 2, byrow = TRUE)

cat("\nMatriz M:\n"); print(M)
cat("det(M):", det(M), "\n")
# det = 2*3 - 1*5 = 1 → M é invertível

M2 <- matrix(c(1, 2,
               2, 4), nrow = 2, byrow = TRUE)
cat("\nMatriz M2 (singular):\n"); print(M2)
cat("det(M2):", det(M2), "→ matriz singular, não invertível\n")

# Determinante de matriz 3x3
A3 <- matrix(c(1, 2, 3,
               0, 4, 5,
               1, 0, 6), nrow = 3, byrow = TRUE)
cat("\ndet(A3):", det(A3), "\n")


# -----------------------------------------------------------------------------
# 11. MATRIZ INVERSA
# -----------------------------------------------------------------------------

cat("\nInversa de M:\n"); print(solve(M))

# Verificando: M %*% solve(M) deve ser a identidade
cat("M %*% solve(M):\n"); print(round(M %*% solve(M), 10))

# Inversa de A3
cat("Inversa de A3:\n"); print(round(solve(A3), 4))


# -----------------------------------------------------------------------------
# 12. MATRIZ IDENTIDADE
# -----------------------------------------------------------------------------

I3 <- diag(3)   # identidade 3x3
cat("\nMatriz Identidade 3x3:\n"); print(I3)

# Propriedade: A %*% I = A
cat("A3 %*% I3 == A3?\n")
cat(all.equal(A3 %*% I3, A3), "\n")


# -----------------------------------------------------------------------------
# 13. MATRIZ DIAGONAL
# -----------------------------------------------------------------------------

D_diag <- diag(c(3, 7, 2))
cat("\nMatriz Diagonal:\n"); print(D_diag)

# Inversa de matriz diagonal: basta inverter os elementos
cat("Inversa de D_diag:\n"); print(solve(D_diag))
cat("(equivalente a diag(1/c(3,7,2))):\n"); print(diag(1/c(3, 7, 2)))


# -----------------------------------------------------------------------------
# 14. MATRIZ SIMÉTRICA
# -----------------------------------------------------------------------------

# Uma matriz é simétrica se A = t(A)
S_mat <- matrix(c(4, 2, 1,
                  2, 5, 3,
                  1, 3, 6), nrow = 3, byrow = TRUE)

cat("\nMatriz Simétrica S:\n"); print(S_mat)
cat("S == t(S)?", all.equal(S_mat, t(S_mat)), "\n")

# A matriz de covariância é sempre simétrica
dados <- matrix(rnorm(50), nrow = 10)
cov_mat <- cov(dados)
cat("Matriz de covariância é simétrica?",
    all.equal(cov_mat, t(cov_mat)), "\n")


# -----------------------------------------------------------------------------
# 15. AUTOVALORES E AUTOVETORES
# -----------------------------------------------------------------------------

# Para A v = lambda v: v é autovetor, lambda é autovalor
cat("\n--- Autovalores e Autovetores ---\n")

A_eig <- matrix(c(3, 1,
                  1, 3), nrow = 2, byrow = TRUE)

resultado <- eigen(A_eig)

cat("Matriz A:\n"); print(A_eig)
cat("Autovalores:\n"); print(resultado$values)
cat("Autovetores (colunas):\n"); print(resultado$vectors)

# Verificando: A %*% v = lambda * v
v1     <- resultado$vectors[, 1]
lambda1 <- resultado$values[1]
cat("\nVerificação A %*% v1 == lambda1 * v1:\n")
cat("A %*% v1:   ", A_eig %*% v1, "\n")
cat("lambda1*v1: ", lambda1 * v1, "\n")

# Autovalores de uma matriz de covariância
cat("\nAutovalores da matriz de covariância:\n")
print(eigen(S_mat)$values)
cat("(todos positivos → S é positiva definida)\n")


# -----------------------------------------------------------------------------
# 16. DECOMPOSIÇÃO ESPECTRAL
# -----------------------------------------------------------------------------

# A = P %*% diag(lambda) %*% t(P)
# onde P = matriz de autovetores, lambda = autovalores

cat("\n--- Decomposição Espectral ---\n")
P      <- resultado$vectors
Lambda <- diag(resultado$values)

A_reconstruida <- P %*% Lambda %*% t(P)
cat("A original:\n");       print(A_eig)
cat("A reconstruída via P Lambda P':\n"); print(round(A_reconstruida, 10))


# -----------------------------------------------------------------------------
# 17. MATRIZ POSITIVA DEFINIDA E SEMIDEFINIDA
# -----------------------------------------------------------------------------

cat("\n--- Classificação de Matrizes por Autovalores ---\n")

# Positiva Definida: todos autovalores > 0
PD <- matrix(c(4, 2,
               2, 3), nrow = 2, byrow = TRUE)
cat("PD — autovalores:", eigen(PD)$values,
    "→ todos > 0, positiva definida\n")

# Positiva Semidefinida: todos autovalores >= 0 (pelo menos um = 0)
PSD <- matrix(c(1, 1,
                1, 1), nrow = 2, byrow = TRUE)
cat("PSD — autovalores:", round(eigen(PSD)$values, 10),
    "→ um = 0, positiva semidefinida\n")

# Indefinida: autovalores de sinais mistos
INDEF <- matrix(c(1,  2,
                  2, -3), nrow = 2, byrow = TRUE)
cat("INDEF — autovalores:", round(eigen(INDEF)$values, 4),
    "→ sinais mistos, indefinida\n")

# Verificação formal de positiva definida via Cholesky
cat("\nCholesky de PD (funciona se PD for positiva definida):\n")
tryCatch(print(chol(PD)), error = function(e) cat("FALHOU\n"))

cat("Cholesky de PSD:\n")
tryCatch(print(chol(PSD)), error = function(e) cat("FALHOU — não é PD\n"))




S <- matrix(c(2, 4, 8,
              4, 8, 16,
              12, 24, 48), nrow = 3, byrow = TRUE)
solve(S)

lambda_1 = 0.05
m_I = matrix(c(1, 0, 0,
                    0, 1, 0,
                    0, 0, 1), nrow = 3, byrow = TRUE)
m_I

solve(S + lambda_1*m_I)







