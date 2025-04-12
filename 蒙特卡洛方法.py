import numpy as np
import random
P = np.array([
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 0.0],
])
R = np.array([
    [-1.0, 0.0, -100.0, -100.0, -100.0],
    [-1.0, -100.0, -2.0, -100.0, -100.0],
    [-100.0, -100.0, -100.0, -2.0, 0.0],
    [-100.0, 1.0, 1.0, 1.0, 10.0],
    [-100.0, -100.0, -100.0, -100.0, -100.0],
])
def get_chain(max_len):
    ss=[]
    rs=[]
    s=random.choice(range(4))
    ss.append(s)
    for _ in range(max_len):
        s_next=np.random.choice(np.arange(5),p=P[s])
        r=R[s,s_next]
        s=s_next
        ss.append(s)
        rs.append(r)
        if s==4:
            break
    return ss,rs

def get_chains(N,max_len):
    ss=[]
    rs=[]
    for _ in range(N):
        s,r=get_chain(max_len)
        ss.append(s)
        rs.append(r)
    return ss,rs

def get_vaule(rs):
    