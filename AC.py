import gym
from matplotlib import pyplot as plt
import torch
import random
from IPython import display
class MyWrapper(gym.Wrapper):#初始化环境
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

def show():#展示
    plt.imshow(env.render())
    plt.show()

model = torch.nn.Sequential(#定义策略网络用来估计概率
    torch.nn.Linear(4, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 2),
    torch.nn.Softmax(dim=1),
)
model_td = sequential = torch.nn.Sequential(#价值网络估计动作价值
    torch.nn.Linear(4, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1),
)

def get_action(state):#获取动作
    state = torch.FloatTensor(state).reshape(1, 4)
    prob = model(state)
    action = random.choices(range(2), weights=prob[0].tolist(), k=1)[0]
    return action

def get_data():#获取数据
    states=[]
    rewards=[]
    actions=[]
    next_states=[]
    overs=[]
    state=env.reset()
    over=False
    while not over:
        action=get_action(state)
        next_state,reward,over,_=env.step(action)
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        next_states.append(next_state)
        overs.append(over)
        state=next_state
    states = torch.FloatTensor(states).reshape(-1, 4)
    rewards = torch.FloatTensor(rewards).reshape(-1, 1)
    actions = torch.LongTensor(actions).reshape(-1, 1)
    next_states = torch.FloatTensor(next_states).reshape(-1, 4)
    overs = torch.LongTensor(overs).reshape(-1, 1)
    return states, rewards, actions, next_states, overs

get_data()
def test(play):#展示
    state = env.reset()
    reward_sum = 0
    over = False
    while not over:
        action = get_action(state)
        state, reward, over, _ = env.step(action)
        reward_sum += reward
        if play and random.random() < 0.2: 
            display.clear_output(wait=True)
            show()
    return reward_sum

def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)#初始化优化器和损失函数
    optimizer_td = torch.optim.Adam(model_td.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    for i in range(1000):
        states,rewards,actions,next_states,overs=get_data()
        values=model_td(states)#计算价值
        targets=model_td(next_states)#计算Q值
        targets*=(1-overs)
        targets+=rewards
        delta=(targets-values).detach()
        probs=model(states)
        probs=probs.gather(dim=1,index=actions)
        loss=(-probs.log()*delta).mean()#策略梯度更新策略
        loss_td=loss_fn(values,targets.detach())#计算TD损失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_td.zero_grad()
        loss_td.backward()
        optimizer_td.step()
        if i % 100 == 0:
            test_result = sum([test(play=False) for _ in range(10)]) / 10
            print(i, test_result)
train()