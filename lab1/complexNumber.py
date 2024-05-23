class ComplexNumber:
    def __init__(self, real_number, imaginary_unit):
        self.real_number = real_number
        self.imaginary_unit = imaginary_unit

    def __repr__(self) -> str:
        if self.imaginary_unit > 0:
            repr = f"{self.real_number}+{self.imaginary_unit}i"
        elif self.imaginary_unit == 0:
            repr = f"{self.real_number}"
        else:
            repr = f"{self.real_number}{self.imaginary_unit}i"
        return repr
    
    def __add__(self, other):
        return ComplexNumber(self.real_number + other.real_number, self.imaginary_unit + other.imaginary_unit)
    
    def __sub__(self, other):
        return ComplexNumber(self.real_number - other.real_number, self.imaginary_unit - other.imaginary_unit)
    
    def __mul__(self, other):
        rn = self.real_number * other.real_number - self.imaginary_unit * other.imaginary_unit
        iu = self.real_number * other.imaginary_unit + self.imaginary_unit * other.real_number
        return ComplexNumber(rn, iu)
    
    def help(self):
        print(f"Complex number \n\t real part: {self.real_number} \n\t imaginary unit: {self.imaginary_unit}")


number1 = ComplexNumber(5, 2)
number2 = ComplexNumber(3, -7)
number3 = ComplexNumber(-4, 5)
number4 = ComplexNumber(-1, -2)

print(number1)
print(number2)
print(number3)
print(number4)

sum = number1 + number2
print(sum)

sub = number3 - number2
print(sub)

mul = number1 * number2
print(mul)

number1.help()
