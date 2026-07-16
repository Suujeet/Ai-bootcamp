count = int(input("How many Fibonacci terms? "))
a, b = 0, 1

for _ in range(count):
    print(a, end=" ")
    a, b = b, a + b
print()
