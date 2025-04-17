import gym
from matplotlib import pyplot as plt
class MyWrapper(gym.Wrapper):
    def __init__(self):
        env=gym.make('CartPole-v1',render_mode='rgb_array')
        super().__init__(env)
        self.env=env
        self.step_n=0

    def reset(self):
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
show()

