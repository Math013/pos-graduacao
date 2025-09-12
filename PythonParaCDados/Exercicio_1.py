ano_nascimento = int(input("Digite o ano de nascimento: "))
ano_atual = int(input("Digite o ano atual: "))
ano_futuro = 2050
idade_atual = ano_atual - ano_nascimento
idade_futuro = ano_futuro - ano_nascimento

if idade_atual <= 0:
    print("Refaça e insira os anos de forma correta")
else:
    print(f"Sua idade atual: {idade_atual}")

if idade_futuro <= 0:
    print("Refaça e insira os anos de forma correta")
else:
    print(f"Sua idade em 2050 será: {idade_futuro}")
