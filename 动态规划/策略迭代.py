import numpy as np
def get_state(row,col):#定义表格
    if row != 3:
        return 'ground'
    if row == 3 and col == 0:
        return 'ground'
    if row == 3 and col == 11:
        return 'terminal'
    return 'trap'

def move(row,col,action):#移动
    if get_state(row, col) in ['trap', 'terminal']: #如果当前已经在陷阱或者终点，则不能执行任何动作，反馈都是0
        return row, col, 0
    if action == 0:#↑
        row -= 1
    if action == 1: #↓
        row += 1
    if action == 2: #←
        col -= 1
    if action == 3: #→
        col += 1
    row = max(0, row) #不允许走到地图外面去
    row = min(3, row)
    col = max(0, col)
    col = min(11, col)
    reward = -1#是陷阱的话，奖励是-100，否则都是-1
    if get_state(row, col) == 'trap':
        reward = -100
    return row, col, reward

values=np.zeros([4,12])
pi=np.ones([4,12,4])*0.25

def get_qsa(row,col,action):#获得每一个动作的价值
    next_row,next_col,reward=move(row,col,action)
    value=values[next_row,next_col]*0.9
    if get_state(next_row, next_col) in ['trap', 'terminal']:
        value = 0
    return reward+value

def get_v():#获得每一个状态的价值
    new_values=np.zeros([4,12])
    for row in range(4):
        for col in range(12):
            action_value=np.zeros(4)
            for action in range(4):
                action_value[action]=get_qsa(row,col,action)
            action_value*=pi[row,col]
            new_values[row,col]=action_value.sum()
    return new_values

def get_pi():#更新概率矩阵
    new_pi=np.zeros([4,12,4])
    for row in range(4):
        for col in range(12):
            action_value=np.zeros(4)
            for action in range(4):
                action_value[action]=get_qsa(row,col,action)
            count=(action_value==action_value.max()).sum()
            for action in range(4):
                if(action_value[action]==action_value.max()):
                    new_pi[row,col,action]=1/count
                else:
                    new_pi[row,col,action]=0
    return new_pi

for _ in range(10):#迭代更新
    for _ in range(100):
        values=get_v()
    pi=get_pi()

for row in range(4):#打印迭代后的状态
    line = ''
    for col in range(12):
        action = pi[row, col].argmax()
        action = {0: '↑', 1: '↓', 2: '←', 3: '→'}[action]
        line += action
    print(line)
        