import gym
from matplotlib import pyplot as plt
import torch
import random
from IPython import display
class MyWrapper(gym.Wrapper):#继承Wrapper
    def __init__(self):
        env=gym.make('CartPole-v1',render_mode='rgb_array')
        super().__init__(env)#调用父类初始化
        self.env=env#保存环境引用
        self.step_n=0#初始化计数器

    def reset(self):#重置环境
        state,_=self.env.reset()
        self.step_n=0
        return state
    
    def step(self,action):
        state,reward, terminated, truncated, info = self.env.step(action)
        done=terminated or truncated
        self.step_n+=1
        if self.step_n>200:
            done=True
        return state,reward,done,info

def show():
    plt.imshow(env.render())
    plt.show()

env=MyWrapper()
env.reset()
#show()

def test_env():
    state=env.reset()
    print("state=",state)
    print("action_space=",env.action_space)
    action=env.action_space.sample()
    print("action=",action)
    state,reward,over,_=env.step(action)
    print("state=",state)
    print("reward=",reward)
    print("over=",over)

model=torch.nn.Sequential(
    torch.nn.Linear(4,128),
    torch.nn.ReLU(),
    torch.nn.Linear(128,2)
)

def get_action(state):
    if random.random()<0.01:
        return random.choice([0,1])
    state=torch.FloatTensor(state).reshape(1,4)
    return model(state).argmax().item()
#print(get_action([0.0013847, -0.01194451, 0.04260966, 0.00688801]))

data=[]
def update_data():
    old_count=len(data)
    while len(data)-old_count<200:
        state=env.reset()
        over=False
        while not over:
            action=get_action(state)
            next_state,reward,over,_=env.step(action)
            data.append((state,action,reward,next_state,over))
            state=next_state
    update_count=len(data)-old_count
    drop_count=max(len(data)-10000,0)
    while len(data)>10000:
        data.pop(0)
    return update_count,drop_count

def get_sample():
    samples=random.sample(data,64)
    state = torch.FloatTensor([i[0] for i in samples])
    action = torch.LongTensor([i[1] for i in samples])
    reward = torch.FloatTensor([i[2] for i in samples])
    next_state = torch.FloatTensor([i[3] for i in samples])
    over = torch.LongTensor([i[4] for i in samples])
    return state, action, reward, next_state, over

#update_data()
#state, action, reward, next_state, over = get_sample()
#print(state[:5], action, reward, next_state[:5], over)

def get_value(state,action):
    value=model(state)
    value=value[range(64),action]
    return value

def get_target(reward,next_state,over):
    with torch.no_grad():
        target=model(next_state)
    target=target.max(dim=1)[0]
    for i in range(64):
        if over[i]:
            target[i]=0
    target*=0.98
    target+=reward
    return target

def test(play):
    state=env.reset()
    reward_sum=0
    over=False
    while not over:
        action=get_action(state)
        state,reward,over,_=env.step(action)
        reward_sum+=reward
        if play:
            display.clear_output(wait=True)
            show()
    return reward_sum

def train():
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(500):
        update_count, drop_count = update_data() #更新N条数据
        for i in range(200): #每次更新过数据后,学习N次
            state, action, reward, next_state, over = get_sample() #采样一批数据
            value = get_value(state, action)#计算一批样本的value和target
            target = get_target(reward, next_state, over)
            loss = loss_fn(value, target)#更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            test_result = sum([test(play=False) for _ in range(20)]) / 20
            print(epoch, len(data), update_count, drop_count, test_result)
train()
print(test(play=False))