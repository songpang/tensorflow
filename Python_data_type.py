# Python_data_type.py

# 단축키
# Ctrl + / : 주석문 처리
# Ctrl + shift + F10 : 새로 작성된 소스를 실행
# shift + F10 : 현재 작업중인 소스를 실행


a = 12
b = 3.14
a, b = 12, 3.14
print(a+b)

c = 12, 3.14
print(c)
print(c[0])
print(c[1])


print('-'*30)

# 연산자(operator) : 산술, 관계, 논리
# 산술 : + - * / //(원래 알던 나누기) **(^2) %

a, b = 13, 6
print(a+b)
print(a-b)
print(a*b)
print('-'*30)
print(a/b)  # 2.16666... 일반적으로 생각하는 나눗셈
print(a//b) # 2, 정수 몫을 구하는 나눗셈
print(a**b)
print(a%b)

# 문제 :두 자리 양수를 거꾸로 뒤집어보세요
# 29 -- > 92

n = 29
n = n//10 + (n%10)*10

print(n)
print('-'*30)

print(a,b)
print(a>b)
print(a>=b)
print(a<b)
print(a<=b)
print(a==b)
print(a!=b)

# 논리     :  and or not
# 다른언어 :    && || !
print('-'*30)
print(True and True)
print(True and False)
print(True or False)
print(False and False)

# data type
# List : [],      mutable

a = [1,3,5,7]
print(a[0], a[1], a[2], a[3], a[-1]) # indexing
print(a[1:3])                        # slicing
a[0] = 10
a.append(8)
print(a)

print('-'*30)
# tuple : ( ), immutable
a = (1,3,5,7) # == a = 1,3,5,7
print(a[0], a[1], a[2], a[3], a[-1])
print(a[1:3])
#a[0] = 10        # ERROR !

print('-'*30)
# dictinary : { }
d = { 'name':'kildong', 'age':40}
print(d['name'], d['age'])

# for loop
for i in range(1,11) :
    print(i, end=' ') # end = ' ' -> 옆으로 출력하게 하는 방법

print()
for i in range(1,11, 2) :
    print(i, end=' ')

print()
for i in range(11,0, -1) :
    print(i, end=' ')

print()
for i in range(11) :
    print(i, end=' ')

print()
print('-'*30)


# 문제 0부터 99까지의 정수를 한줄에 10개씩 출력해보시오.
for i in range(100) :
    print(i, end = ' ')

    if i%10 == 9 :
# function : 함수
        print()

# 반환값이 없고 매개변수(인자) 없고
def f_1() :
    print('f_1 is called')

f_1()

# 반환값이 없고 매개변수 있고
def f_2(a,b) :
    print('f_2', a, b, a+b)

f_2(12,34)

def f_22(a1,a2,a3='aaa') :
    print('f_22', a1, a2, a3)

f_2(12,34)
f_2('hello', 'knu')

f_22('hello', 'knu')

print('-'*30)
def f_3():
    print('f_3')
    return 78

print(f_3())

def f_4(a,b):
    c = a+b
    return c

print(f_4(3,5))

# 문제 : 2개의 정수에 대해 큰 수, 작은 수의 순서로 반환하는 함수를 만드세요.
# 3, 5 --> 3, 5
# 5, 3 --> 3, 5

def order(a,b) :
    if a>b:
        print(b, a)
    else: print(a,b)

def order2(a,b):
    if a>b:
        c = b, a
    return c;

order(352,10)
print(order2(352,10))



