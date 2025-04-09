import numpy as np
import random
probs=np.random.uniform(size=10)
rewards=[[1]for _ in range(10) ]
def chose_one():#选择一个老虎机
    if random.random()<0.01:
        return random.randint(0,9)
    rewards_mean=[np.mean(i)for i in rewards]
    return np.argmax(rewards_mean)

def try_play():#进行试玩
    i=chose_one()
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