# cost_gradient.py

import matplotlib.pyplot as plt

def cost(x,y,w):
    c = 0
    for i in range(len(x)): # 입력 x만큼 loop를 돌린다.
        hx = w * x[i]
        loss = (hx-y[i])**2
        c += loss
    return c / len(x)


# def test_cost_notused():
#     x = [1,2,3]
#     y = [1,2,3]
#
#     print(cost(x,y,-1))
#     print(cost(x,y,0))
#     print(cost(x,y,1))
#     print(cost(x,y,2))
#     print(cost(x,y,3))
#
#     for i in range(-30,51):
#         w = i/10
#         c = cost(x,y,w)
#
#         print(w, c)
#         plt.plot(w, c, 'ro')
#     plt.show()

# 비용함수의 미분의 평균값
def gradient_decent(x,y,w):
    grad = 0
    for i in range(len(x)):
        hx = w * x[i]
        grad += (hx - y[i])*x[i]
    return grad / len(x)

x = [1,2,3]
y = [1,2,3]

w = 10
old = 100

for i in range(100):
    c = cost(x,y,w)
    grad = gradient_decent(x,y,w)
    w = w - 0.1*grad             # Learning Rate
    print(i, c, w, grad)
    if c>=old and abs (c-old) < 1.0e-15:
        break
    old = c
    plt.plot((0,5),(0,5*w))

print('weight =', w)

plt.plot(x,y,'ro')
plt.xlim(0,5)
plt.ylim(0,5)
plt.show()