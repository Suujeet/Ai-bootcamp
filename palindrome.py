text = input("Enter a word: ").strip().lower()

if text == text[::-1]:
    print(f"{text} is a palindrome.")
else:
    print(f"{text} is not a palindrome.")
