import gym
from matplotlib import pyplot as plt
import torch
import random
from IPython import display
class MyWrapper(gym.Wrapper):#继承Wrapper
    def __init__(self):
        env=gym.make('CartPole-v1',render_mode='rgb_array')#返回图像的数组形式
        super().__init__(env)#调用父类初始化
        self.env=env#保存环境引用
        self.step_n=0#初始化计数器

    def reset(self):#重置环境
        state,_=self.env.reset()#状态重置
        self.step_n=0
        return state
    
    def step(self,action):#步进方法
        state,reward, terminated, truncated, info = self.env.step(action)#传入动作
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

model=torch.nn.Sequential(#创建顺序执行的神经网络容器
    torch.nn.Linear(4,128),#线性全连接层，输入4，隐藏层神经元128
    torch.nn.ReLU(),#激活函数
    torch.nn.Linear(128,2)#全连接层，输入128，输出两个动作
)

def get_action(state):
    if random.random()<0.01:
        return random.choice([0,1])
    state=torch.FloatTensor(state).reshape(1,4)#将数组转换为pytorch中的张量
    return model(state).argmax().item()#item将张量中的标量转换为python数值
#print(get_action([0.0013847, -0.01194451, 0.04260966, 0.00688801]))

data=[]
def update_data():#更新数据
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

def get_sample():#获取数据，经验回放
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

def get_value(state,action):#获得价值
    value=model(state)
    value=value[range(64),action]
    return value

def get_target(reward,next_state,over):#获得Q值
    with torch.no_grad():#禁用梯度计算
        target=model(next_state)
    target=target.max(dim=1)[0]#沿动作维度取最大值
    for i in range(64):
        if over[i]:
            target[i]=0
    target*=0.98
    target+=reward
    return target

def test(play):#游玩游戏
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
    model.train()#开启训练模式
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)#启用优化器，指定参数
    loss_fn = torch.nn.MSELoss()#损失函数：目标Q值和预测Q值的差异
    for epoch in range(500):
        update_count, drop_count = update_data() #更新N条数据
        for i in range(200): #每次更新过数据后,学习N次
            state, action, reward, next_state, over = get_sample() #采样一批数据
            value = get_value(state, action)#计算一批样本的value和target
            target = get_target(reward, next_state, over)
            loss = loss_fn(value, target)#计算损失
            optimizer.zero_grad()#清空梯度
            loss.backward()#反向传播
            optimizer.step()#更新梯度
        if epoch % 50 == 0:
            test_result = sum([test(play=False) for _ in range(20)]) / 20
            print(epoch, len(data), update_count, drop_count, test_result)
train()
print(test(play=False))