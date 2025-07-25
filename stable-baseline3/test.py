import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('CartPole-v1')
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=5_0000, progress_bar=True)
print(evaluate_policy(model, env, n_eval_episodes=20))
model.save('Reinforcement-learning/models')
model=PPO.load('Reinforcement-learning/models/1')