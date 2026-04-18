def count_bits(n):
    ones = 0
    zeros = 0
    
    temp = n
    while temp > 0:
        if temp & 1 == 1:
            ones += 1
        else:
            zeros += 1
        temp >>= 1

    return ones, zeros


def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def analyze_number(n):
    print("\n========== RESULT ==========")
    print("Number:", n)
    print("Binary:", bin(n))
    print("Bit Length:", n.bit_length())

    ones, zeros = count_bits(n)
    print("Number of 1 bits:", ones)
    print("Number of 0 bits:", zeros)

    if n % 2 == 0:
        print("Even or Odd: EVEN")
    else:
        print("Even or Odd: ODD")

    if is_prime(n):
        print("Prime Check: PRIME")
    else:
        print("Prime Check: NOT PRIME")

    print("============================\n")


# Main loop
while True:
    user_input = input("Enter a positive number (or type 'q' to quit): ")

    if user_input.lower() == 'q':
        print("Exiting program. Goodbye!")
        break

    if not user_input.isdigit():
        print("Invalid input! Please enter a positive number.")
        continue

    number = int(user_input)
    analyze_number(number)