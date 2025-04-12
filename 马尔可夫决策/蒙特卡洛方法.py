import numpy as np
import random

P = np.array([
    [0.3, 0.2, 0.2, 0.2, 0.1],  # s=0 稍微倾向于 s=0
    [0.2, 0.3, 0.2, 0.2, 0.1],  # s=1 稍微倾向于 s=1
    [0.1, 0.1, 0.3, 0.4, 0.1],  # s=2 倾向于 s=3
    [0.1, 0.1, 0.1, 0.3, 0.4],  # s=3 倾向于 s=4
    [0.2, 0.2, 0.2, 0.2, 0.2],  # s=4 均匀分布（终止状态）
])

R = np.array([
    [1.0, 1.0, 1.0, 1.0, 1.0],  # s=0 的转移奖励较低
    [1.0, 1.0, 1.0, 1.0, 1.0],  # s=1 的转移奖励较低
    [1.0, 1.0, 1.0, 5.0, 1.0],  # s=2 → s=3 奖励高
    [1.0, 1.0, 1.0, 1.0, 10.0], # s=3 → s=4 奖励最高
    [1.0, 1.0, 1.0, 1.0, 1.0],  # s=4 的转移奖励（终止状态）
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
        sum+=0.5**i*r
    return sum

def monte_carlo(ss,rs):
    values=[[]for _ in range(5)]
    for s,r in zip(ss,rs):
        values[s[0]].append(get_value(r))#计算第一步的价值
    return [np.mean(i)for i in values]#求平均估计出每一步的价值

def optimize_transition_matrix(ss, rs, values, learning_rate=0.1):
    counts = np.zeros((5, 5)) # 初始化计数矩阵和新的转移矩阵
    new_P = np.zeros((5, 5))
    for episode_states in ss: # 统计状态转移次数
        for i in range(len(episode_states)-1):
            current_state = episode_states[i]
            next_state = episode_states[i+1]
            counts[current_state, next_state] += 1
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

# 初始蒙特卡洛估计
initial_ss, initial_rs = get_chains(10000, 200)
initial_values = monte_carlo(initial_ss, initial_rs)
print("初始状态价值估计:", initial_values)

# 第一次优化
optimized_P = optimize_transition_matrix(initial_ss, initial_rs, initial_values)
print("优化后的转移矩阵:")
print(optimized_P)

# 使用优化后的转移矩阵进行新的模拟
P = optimized_P  # 更新全局转移矩阵

# 再次运行蒙特卡洛评估
new_ss, new_rs = get_chains(10000, 200)
new_values = monte_carlo(new_ss, new_rs)
print("优化后的状态价值估计:", new_values)

test_time=25
# 可以多次迭代这个过程
for iteration in range(test_time):
    print(f"\n迭代 {iteration+1}:")
    optimized_P = optimize_transition_matrix(new_ss, new_rs, new_values)
    P = optimized_P
    
    new_ss, new_rs = get_chains(10000, 200)
    new_values = monte_carlo(new_ss, new_rs)
    print(f"迭代 {iteration+1} 状态价值:", new_values)
    print(f"迭代 {iteration+1} 转移矩阵:")
    print(P)