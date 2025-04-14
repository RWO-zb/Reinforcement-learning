import numpy as np
P = np.array([#概率矩阵
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
])
R = np.array([-1, -2, -2, 10, 1, 0])#奖励
def get_value(chain):#求奖励
    s=0
    for i,c in enumerate(chain):
        s+=R[c]*0.5**i
    return s

def get_bellman1():#迭代法
    values=np.ones([6])
    for _ in range(200):
        for i in range(6):
            values[i]=R[i]+0.5*P[i].dot(values)
    return values

def get_bellman2():#直接求矩阵逆
    mat=np.eye(*P.shape)
    mat-=0.5*P
    mat = np.linalg.inv(mat)
    return mat.dot(R)

print(get_bellman1())
print(get_bellman2())

