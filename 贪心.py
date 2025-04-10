import numpy as np
import random
probs=np.random.uniform(size=10)
rewards=[[1]for _ in range(10) ]
def chose_one1():#选择一个老虎机
    if random.random()<0.01:
        return random.randint(0,9)
    rewards_mean=[np.mean(i)for i in rewards]
    return np.argmax(rewards_mean)

def chose_one2():#按游玩次数让探索率逐渐下降
    play_count=sum([len(i)for i in rewards])
    if random.random()<1/play_count:
        return random.randint(0,9)
    rewards_mean=[np.mean(i)for i in rewards]
    return np.argmax(rewards_mean)

def chose_one3():#上置信界
    play_count=[len(i)for i in rewards]
    play_count=np.array(play_count)
    fenzi = play_count.sum()**0.5
    fenmu = play_count * 2
    ucb = fenzi / fenmu#ucb代表不确定性，随着探索次数增加会急剧减小
    ucb=ucb**0.5
    rewards_mean=[np.mean(i)for i in rewards]
    rewards_mean=np.array(rewards_mean)
    ucb+=rewards_mean
    return np.argmax(ucb)

def chose_one4():#汤普森法直接估计每个老虎机的中奖概率
    count_1=[sum(i)+1 for i in rewards]
    count_0=[sum(1-np.array(i))+1 for i in rewards]
    beta=np.random.beta(count_1,count_0)
    return np.argmax(beta)

def try_play():#进行试玩
    i=chose_one4()
    reward=0
    if random.random()<probs[i]:#进行游玩
        reward=1
    rewards[i].append(reward)

def get_result():
    for _ in range(5000):
        try_play()
    target=probs.max()*5000
    result=sum([sum(i)for i in rewards])
    return target,result
print(get_result())