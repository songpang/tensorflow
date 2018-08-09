# softmax.py
import math


def softmax(a, b, c):
    Ya = math.e**a
    Yb = math.e**b
    Yc = math.e**c

    base = Ya + Yb + Yc
    Pa = Ya/base
    Pb = Yb / base
    Pc = Yc / base

    return Pa, Pb, Pc


print(softmax(2.0, 1.0, 0.1))