import math

X = [1,2,3,4,5]
Y = [0,0,0,1,1]

def sigmoid(z):
    return 1/(1+math.exp(-z))

def cost(w,c):
    total = 0
    for x,y in zip(X,Y):
        p = sigmoid(w*x+c)
        p = min(max(p,1e-12),1-1e-12)
        total += -(y*math.log(p) + (1-y)*math.log(1-p))
    return total

w, c = 0.0, 0.0
eta = 0.1
history = []
for it in range(1, 2001):
    dw = 0
    dc = 0
    for x,y in zip(X,Y):
        p = sigmoid(w*x+c)
        dw += (p-y)*x
        dc += (p-y)
    w -= eta*dw
    c -= eta*dc
    if it in (1,2,3,10,50,200,500,1000,2000):
        print(f'iter {it}: w={w:.4f} c={c:.4f} cost={cost(w,c):.4f}')

print()
print('final predictions:')
for x,y in zip(X,Y):
    p = sigmoid(w*x+c)
    print(f'x={x} y={y} p={p:.4f}')
print('boundary x where wx+c=0:', -c/w)
"
Output

iter 1: w=0.1500 c=-0.0500 cost=3.3420
iter 2: w=0.1177 c=-0.1488 cost=3.2579
iter 3: w=0.1629 c=-0.2241 cost=3.1855
iter 10: w=0.3035 c=-0.7511 cost=2.7638
iter 50: w=0.8681 c=-2.7514 cost=1.6551
iter 200: w=1.7719 c=-5.9666 cost=0.8557
iter 500: w=2.6066 c=-8.9305 cost=0.5290
iter 1000: w=3.4169 c=-11.7916 cost=0.3489
iter 2000: w=4.4117 c=-15.2898 cost=0.2138

final predictions:
x=1 y=0 p=0.0000
x=2 y=0 p=0.0016
x=3 y=0 p=0.1136
x=4 y=1 p=0.9135
x=5 y=1 p=0.9989
boundary x where wx+c=0: 3.465707366002805