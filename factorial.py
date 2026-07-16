number = int(input("Enter a non-negative integer: "))

if number < 0:
    print("Factorial is not defined for negative numbers.")
else:
    result = 1
    for value in range(2, number + 1):
        result *= value
    print(f"{number}! = {result}")
