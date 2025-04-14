import gym
from matplotlib import pyplot as plt
import numpy as np
from IPython import display
import time
env = gym.make('FrozenLake-v1',#创建环境
               render_mode='rgb_array',
               is_slippery=False,#不会滑
               map_name='4x4',#尺寸
               desc=['SFFF', 'FHFH', 'FFFH', 'HFFG'])#地形
env.reset()
env = env.unwrapped#解封装才能访问

def show():#打印
    plt.imshow(env.render())
    plt.show()

values=np.zeros(16)
pi=np.ones([16,4])*0.25

def get_qsa(state,action):#获得每个动作的价值
    value=0.0
    for prop, next_state, reward, over in env.P[state][action]:
       next_value=values[next_state]*0.9
       if over:
           next_value=0
       next_value+=reward
       next_value*=prop
       value+=next_value
    return value

def get_value():#求每个状态的价值
    new_values=np.zeros([16])
    for state in range(16):
        action_value=np.zeros(4)
        for action in range(4):
            action_value[action]=get_qsa(state,action)
        new_values[state]=action_value.max()#价值迭代
    return new_values

def get_pi():#迭代求概率矩阵
    new_pi=np.zeros([16,4])
    for state in range(16):
        action_value=np.zeros(4)
        for action in range(4):
            action_value[action]=get_qsa(state,action)
        count=(action_value==action_value.max()).sum()
        for action in range(4):
            if action_value[action]==action_value.max():
                new_pi[state][action]=1/count
            else:
                new_pi[state][action]=0
    return new_pi

for _ in range(10):#迭代
    for _ in range(100):
        values = get_value()
    pi = get_pi()

def print_pi():
    for row in range(4):
        line = ''
        for col in range(4):
            state = row * 4 + col
            if (row == 1 and col == 1) or (row == 1 and col == 3) or (
                    row == 2 and col == 3) or (row == 3 and col == 0):
                line += '○'
                continue
            if row == 3 and col == 3:
                line += '❤'
                continue
            line += '←↓→↑'[pi[state].argmax()]
        print(line)

def play():#游玩
    env.reset()
    index = 0#起点在0
    for i in range(200):
        action = np.random.choice(np.arange(4), size=1, p=pi[index])[0]
        index, reward, terminated, truncated, _ = env.step(action)
        display.clear_output(wait=True)
        time.sleep(0.1)
        show()
        if terminated or truncated:
            break
    print(i)

play()