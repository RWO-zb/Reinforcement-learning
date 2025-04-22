import gym
from matplotlib import pyplot as plt
import torch
import random
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

model = torch.nn.Sequential(#计算动作的模型,也是真正要用的模型
    torch.nn.Linear(4, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 2),
)
next_model = torch.nn.Sequential(#经验网络,用于评估一个状态的分数
    torch.nn.Linear(4, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 2),
)
next_model.load_state_dict(model.state_dict())#把model的参数复制给next_model

def get_action(state):#获得动作
    if random.random() < 0.01:
        return random.choice([0, 1])
    state = torch.FloatTensor(state).reshape(1, 4)
    return model(state).argmax().item()

datas = []
def update_data():#更新数据
    old_count = len(datas)
    while len(datas) - old_count < 200:#玩到新增了N个数据为止
        state = env.reset()#初始化游戏
        over = False#玩到游戏结束为止
        while not over:
            action = get_action(state)#根据当前状态得到一个动作
            next_state, reward, over, _ = env.step(action)#执行动作,得到反馈
            datas.append((state, action, reward, next_state, over))#记录数据样本
            state = next_state#更新游戏状态,开始下一个动作
    update_count = len(datas) - old_count
    drop_count = max(len(datas) - 10000, 0)
    while len(datas) > 10000:#数据上限,超出时从最老的开始删除
        datas.pop(0)
    return update_count, drop_count
update_data()

def get_sample():#获取一批数据样本
    samples = random.sample(datas, 64)#从样本池中采样
    state = torch.FloatTensor([i[0] for i in samples]).reshape(-1, 4)
    action = torch.LongTensor([i[1] for i in samples]).reshape(-1, 1)
    reward = torch.FloatTensor([i[2] for i in samples]).reshape(-1, 1)
    next_state = torch.FloatTensor([i[3] for i in samples]).reshape(-1, 4)
    over = torch.LongTensor([i[4] for i in samples]).reshape(-1, 1)
    return state, action, reward, next_state, over

state, action, reward, next_state, over = get_sample()

def get_value(state, action):#获得价值
    value = model(state)
    value = value.gather(dim=1, index=action)
    return value

def get_target(reward, next_state, over):#获得Q值
    with torch.no_grad():
        target = next_model(next_state)
    target = target.max(dim=1)[0]
    target = target.reshape(-1, 1)
    target *= 0.98
    target *= (1 - over)
    target += reward
    return target

def test(play):#试玩
    state = env.reset()
    reward_sum = 0
    over = False
    while not over:
        action = get_action(state)
        state, reward, over, _ = env.step(action)
        reward_sum += reward
    return reward_sum

def train():#训练
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = torch.nn.MSELoss()
    for epoch in range(500):
        update_count, drop_count = update_data()
        for i in range(200):
            state, action, reward, next_state, over = get_sample()
            value = get_value(state, action)
            target = get_target(reward, next_state, over)
            loss = loss_fn(value, target)#用损失函数拟合逼近
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                next_model.load_state_dict(model.state_dict())

        if epoch % 50 == 0:
            test_result = sum([test(play=False) for _ in range(20)]) / 20
            print(epoch, len(datas), update_count, drop_count, test_result)

train()