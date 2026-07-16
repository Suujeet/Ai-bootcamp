number = int(input("Enter a positive integer: "))

if number < 2:
    is_prime = False
else:
    is_prime = all(number % divisor != 0 for divisor in range(2, int(number ** 0.5) + 1))

if is_prime:
    print(f"{number} is prime.")
else:
    print(f"{number} is not prime.")
