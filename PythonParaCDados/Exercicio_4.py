n1 = int(input("Digite o primeiro número: "))
n2 = int(input("Digite o segundo número: "))

if n1 > n2:
    for num in range(n2, n1 + 1):
        print(num)
    
elif n2 > n1:
    for num in range(n1, n2 + 1):
        print(num)
    
    #print(f"O maior número é: {n2}")
else:
    print("Os números são iguais.")