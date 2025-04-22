import gym
from matplotlib import pyplot as plt
import torch
import random
from IPython import display
class MyWrapper(gym.Wrapper):
    def __init__(self):
        env = gym.make('CartPole-v1', render_mode='rgb_array')
        super().__init__(env)
        self.env = env
        self.step_n = 0
    def reset(self):
        state, _ = self.env.reset()
        self.step_n = 0
        return state
    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.step_n += 1
        if self.step_n >= 200:
            done = True
        return state, reward, done, info

env = MyWrapper()
env.reset()

def show():
    plt.imshow(env.render())
    plt.show()

model = torch.nn.Sequential(#定义模型
    torch.nn.Linear(4, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 2),
    torch.nn.Softmax(dim=1),
)

def get_action(state):#获取动作
    state=torch.FloatTensor(state).reshape(1,4)
    prob=model(state)
    action = random.choices(range(2), weights=prob[0].tolist(), k=1)[0]#根据概率挑选
    return action

def get_data():#获取一局的数据
    states=[]
    rewards=[]
    actions=[]
    state=env.reset()
    over=False
    while not over:
        action=get_action(state)
        next_state,reward,over,_=env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state=next_state
    return states,rewards,actions

def test(play):
    state = env.reset()#初始化游戏
    reward_sum = 0 #记录反馈值的和,这个值越大越好
    over = False#玩到游戏结束为止
    while not over:#根据当前状态得到一个动作
        action = get_action(state)#执行动作,得到反馈
        state, reward, over, _ = env.step(action)
        reward_sum += reward
        if play and random.random() < 0.2:  #跳帧
            display.clear_output(wait=True)
            show()
    return reward_sum

def train():#训练
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(3000):
        states,rewards,actions=get_data()
        optimizer.zero_grad()
        reward_sum=0
        for i in reversed(range(len(states))):#倒置求值
            reward_sum*=0.98
            reward_sum+=rewards[i]
            state=torch.FloatTensor(states[i]).reshape(1,4)
            prob=model(state)
            prob=prob[0,actions[i]]
            loss=-prob.log()*reward_sum#损失函数
            loss.backward(retain_graph=True)#梯度累计保持计算图
        optimizer.step()

        if epoch % 100 == 0:
            test_result = sum([test(play=False) for _ in range(10)]) / 10
            print(epoch, test_result)

train()