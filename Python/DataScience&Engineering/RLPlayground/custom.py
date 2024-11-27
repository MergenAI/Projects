import gym 
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

disc=Discrete(3)
box=Box(0,1,shape=(3,))
# for i in range(5):
#     print(disc.sample())
#     print(box.sample())

# multi_b=MultiBinary(4)
# # MultiDiscrete([5,2,2]) == bu tanımda her eleman,bulunduğu kümenin en büyük elemanıdır
# tuple=Tuple((disc,box,multi_b))
# print(tuple.sample())
# dict=Dict({"height":Discrete(2),"speed":Box(0,100,shape=1,),"color":multi_b})

class ShowerEnv(Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(3)
        # Temperature array
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # Set start temp
        self.state = 38 + random.randint(-3,3)
        # Set shower length
        self.shower_length = 60
    def step(self, action):
        self.state+=action-1
        self.shower_length-=1
        if self.state>=37 and self.state<=39:
            reward=1
        else:
            reward= -1
        if self.shower_length<=0:
            done=True
        else:
            done=False
        info={}
        return self.state,reward,done,info
    def reset(self):
        self.state=np.array([38 + random.randint(-3,3)]).astype(float)
        self.shower_length=60
        return self.state
    def render(self,mode=None ):
        pass

env=ShowerEnv()
tekrar=5
for bölüm in range(1,tekrar+1):
    durum=env.reset()
    devam=False
    puan=0
    while not devam:
        env.render()
        hareket=env.action_space.sample()
        nstate,ödül,devam,bilgi=env.step(hareket)
        puan+=ödül
    print("Bölüm={}   Puan={}   Durum={}".format(bölüm,puan,durum))


# döküm_kayıt=os.path.join("Eğit","Döküm","custom")
tapçan_kayıt=os.path.join("Eğit","Tapçan","customRL.zip")
# tapçan=PPO("MlpPolicy",env,verbose=1,tensorboard_log=döküm_kayıt)
# tapçan.learn(total_timesteps=50000)
# tapçan.save(tapçan_kayıt)

# tapçan=PPO.load(tapçan_kayıt)
# print(evaluate_policy(tapçan, env, n_eval_episodes=10, render=True))
