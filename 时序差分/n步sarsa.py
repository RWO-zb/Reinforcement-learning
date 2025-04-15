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
state_list=[]
reward_list=[]
action_list=[]

def get_action(row,col):#获得下一个动作
    if random.random()<0.1:#一直随机导致无法完全收敛
        return random.choice(range(4))
    return  Q[row,col].argmax()

def get_update_list(next_row,next_col,next_action):
    target=Q[next_row,next_col,next_action]
    target_list=[]
    for i in range(5):#计算下一步的价值
        taeget=0.9*target+reward_list[i]
        target_list.append(target)
    target_list=list(reversed(target_list))
    value_list=[]
    for i in range(5):#计算当前步的价值
        row,col=state_list[i]
        action=action_list[i]
        value_list.append(Q[row][col][action])
    update_list=[]
    for i in range(5):#计算差分
        update=target_list[i]-value_list[i]
        update*=0.1
        update_list.append(update)
    return update_list

def train():
    for epoch in range(1500):
        row=random.choice(range(4))
        col=0
        action=get_action(row,col)
        reward_sum=0
        state_list.clear()
        reward_list.clear()
        action_list.clear()
        while get_state(row,col) not in ['terminal', 'trap']:
            next_row,next_col,reward=move(row,col,action)
            reward_sum+=reward
            next_action=get_action(next_row,next_col)
            state_list.append([row,col])
            reward_list.append(reward)
            action_list.append(next_action)
            if len(state_list)==5:
                update_list=get_update_list(next_row,next_col,next_action)
                row,col=state_list[0]
                update=update_list[0]
                action=action_list[0]
                Q[row][col][action]+=update
                state_list.pop(0)
                action_list.pop(0)
                reward_list.pop(0)
            row=next_row
            col=next_col
            action=next_action
        for i in range(len(state_list)):
            row,col=state_list[i]
            action=action_list[i]
            update=update_list[i]
            Q[row][col][action]+=update
        if epoch%100==0:
            print(epoch,reward_sum)

train()
for row in range(4):
    line = ''
    for col in range(12):
        action = Q[row, col].argmax()
        action = {0: '↑', 1: '↓', 2: '←', 3: '→'}[action]
        line += action
    print(line)