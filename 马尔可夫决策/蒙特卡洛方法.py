import numpy as np
import random

P = np.array([
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.2], 
    [0.2, 0.2, 0.2, 0.2, 0.2],
    [0,0,0,0,1],#不能给终点随机概率，否则会难以收敛
])

R = np.array([#结果偏向终点
    [0, 0, 0, 0, 100], 
    [0, 0, 0, 0, 100], 
    [0, 0, 0, 0, 100],  
    [0, 0, 0, 0, 100], 
    [0, 0, 0, 0, 100], 
])

def get_chain(max_len):#进行一局n步的探索
    ss=[]
    rs=[]
    s=random.choice(range(5))#随机挑选一个状态开始
    ss.append(s)
    for _ in range(max_len):
        s_next=np.random.choice(np.arange(5),p=P[s])#根据状态矩阵挑选下一个状态
        r=R[s,s_next]
        s=s_next
        ss.append(s)
        rs.append(r)
        if s==4:
            break
    return ss,rs

def get_chains(N,max_len):#进行N局探索
    ss=[]
    rs=[]
    for _ in range(N):
        s,r=get_chain(max_len)
        ss.append(s)
        rs.append(r)
    return ss,rs

def get_value(rs):#统计奖励
    sum=0
    for i,r in enumerate(rs):
        sum+=0.9**i*r
    return sum

def monte_carlo(ss,rs):
    values=[[]for _ in range(5)]
    for s,r in zip(ss,rs):
        values[s[0]].append(get_value(r))#计算第一步的价值
    return [np.mean(i)for i in values]#求平均估计出每一步的价值

def optimize(ss, rs, values, learning_rate=0.01):
    counts = np.zeros((5, 5)) # 初始化计数矩阵和新的转移矩阵
    new_P = np.zeros((5, 5))
    for es in ss: # 统计状态转移次数
        for i in range(len(es)-1):
            cs = es[i]#当前状态
            ns = es[i+1]#下一状态
            counts[cs, ns] += 1
    for s in range(5): # 计算基于价值的转移概率
        total = 0
        for s_next in range(5): # 计算每个转移的价值加权和
            if counts[s, s_next] > 0:# 价值高的转移会获得更高的概率
                new_P[s, s_next] = np.exp(values[s_next] * learning_rate) * counts[s, s_next]
            else:
                new_P[s, s_next] = 0
        row_sum = np.sum(new_P[s])# 归一化
        if row_sum > 0:
            new_P[s] /= row_sum
        else:
            new_P[s] = P[s]  # 如果没有观察到转移，保持原样
    return new_P

new_ss, new_rs = get_chains(5000, 100)
new_values = monte_carlo(new_ss, new_rs)
test_time=10

for iteration in range(test_time):# 可以多次迭代这个过程
    print(f"\n迭代 {iteration+1}:")
    optimized_P = optimize(new_ss, new_rs, new_values)
    P = optimized_P
    
    new_ss, new_rs = get_chains(5000, 100)
    new_values = monte_carlo(new_ss, new_rs)
    print(f"迭代 {iteration+1} 状态价值:", new_values)
    print(f"迭代 {iteration+1} 转移矩阵:")
    print(P)