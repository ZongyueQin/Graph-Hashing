class myInt:
    def __init__(self, a):
        self.a = a + 1

A = myInt(0)

def build(a):
    print(a)
    return a[0][0]

def turn(a):
    print(a)
    print(A.a+a)
    return A.a + a
