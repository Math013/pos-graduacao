# Estudo Análise Multivariada

# Exercício manual 1
X <- matrix(c(2,4, 4,6, 6,8, 8,10), ncol=2, byrow=TRUE)
colMeans(X)

cov(X)
cor(X)

# Exercicio 2
S <- matrix(c(4,2,2,3), nrow = 2)
x  <- c(3,2);  mu <- c(1,1)
mahalanobis(x, center = mu, cov = S)

t(x - mu) %*% solve(S) %*% (x - mu)

S_inversa <- ((1/determinante) * S)

# Exercicio 3
S <- matrix(c(4,2,2,3), nrow = 2)
eigen(S)         # autovalores e autovetores
sum(diag(S))     # traço = soma de autovalores
det(S)           # determinante = produto de autovalores

lambda1 <- (sum(diag(S)) + ((S[1]-S[4])**2 + 4 * (S[2]*S[3]))**0.5)/2
lambda2 <- (sum(diag(S)) - ((S[1]-S[4])**2 + 4 * (S[2]*S[3]))**0.5)/2


result <- lambda1**2 - 7*lambda2 + 8

lambda1 + lambda2

# Exercicio 4
library(DescTools)
X <- matrix(c(120,80, 125,85, 130,82, 118,78, 122,81),
            ncol = 2, byrow = TRUE)
HotellingsT2Test(X, mu = c(120, 80))

# Dataset teste
# PCA no dataset iris (4 variáveis numéricas)
data(iris)
X <- iris[, 1:4]
pca <- prcomp(X, scale. = TRUE)

# Variância explicada por cada componente
summary(pca)
# Visualização
biplot(pca)
# Scree plot
plot(pca, type = "l")
