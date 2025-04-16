import numpy as np
import random
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

Q=np.zeros([4,12,4])
history = dict()#保存历史数据,键是(row,col,action),值是(next_row,next_col,reward)

def get_action(row,col):#获得动作
    if random.random()<0.1:
        random.choice(range(4))
    return Q[row,col].argmax()

def get_update(row,col,action,reward,next_row,next_col):#求差分
    target=0.9*Q[next_row,next_col].max()
    target+=reward
    value=Q[row,col,action]
    update=target-value
    update*=0.1#0.1是学习率，表示对参数的信任程度
    return update

def q_planning():#反刍历史数据
    for _ in range(20):
        row,col,action=random.choice(list(history.keys()))
        next_row,next_col,reward=history[(row,col,action)]
        update=get_update(row,col,action,reward,next_row,next_col)
        Q[row,col,action]+=update

def train():#训练
    for epoch in range(300):
        row=random.choice(range(4))
        col=0
        action=get_action(row,col)
        reward_sum=0
        while get_state(row, col) not in ['terminal', 'trap']:
            next_row,next_col,reward=move(row,col,action)
            reward_sum+=reward
            next_action=get_action(next_row,next_col)
            update=get_update(row,col,action,reward,next_row,next_col)
            Q[row,col,action]+=update
            history[(row,col,action)]=next_row,next_col,reward#保存历史数据
            q_planning()
            row=next_row
            col=next_col
            action=next_action
        if epoch%20==0:
            print(epoch,reward_sum)

train()
for row in range(4):#打印结果
    line = ''
    for col in range(12):
        action = Q[row, col].argmax()
        action = {0: '↑', 1: '↓', 2: '←', 3: '→'}[action]
        line += action
    print(line)
