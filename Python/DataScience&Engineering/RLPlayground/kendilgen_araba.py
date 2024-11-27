import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

environment_name = "CarRacing-v0"
env = gym.make(environment_name)
print(env.action_space)
print(env.observation_space)
def dene():
    episodes = 5
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print('Episode:{} Score:{}'.format(episode, score))
    env.close()

döküm_kayıt=os.path.join("Eğit","Döküm","araç")
tapçan_kayıt=os.path.join("Eğit","Tapçan","araç.zip")
def eğit_ve_kaydet():
    tapçan=PPO("CnnPolicy",env,verbose=1,tensorboard_log=döküm_kayıt)
    tapçan.learn(total_timesteps=20000)
    tapçan.save(tapçan_kayıt)

# eğit_ve_kaydet()
def yükle_çalıştır():
    tapçan=PPO.load(tapçan_kayıt,env)
    evaluate_policy(tapçan, env, n_eval_episodes=2, render=True)

yükle_çalıştır()