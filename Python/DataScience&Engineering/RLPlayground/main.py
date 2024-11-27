import gym
import stable_baselines3
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

"""
Vectorizing:aynı oyunu birden fazla kez anı anda yürütmektir. Bu sebepten ötürü aynı sürede daha fazla eğitim yapılır 
make_atari_env:stable baselines sınıfından atari oyunlarını samalamaya yarar
VecFrameStack:sarmalanmış atari oyunlarını birleştirir

Aynı anda birden fazla oyun götselleştirilemiyor. dolayısıyla yalnızca 1 oyun yürütülmeli 
"""



def dene():
    ad="Breakout-v0"
    oyun=gym.make(ad)
    tekrar = 5
    for bölüm in range(1, tekrar + 1):
        durum = oyun.reset()
        devam = False
        puan = 0
        while not devam:
            oyun.render()
            hareket = oyun.action_space.sample()
            nstate, ödül, devam, bilgi = oyun.step(hareket)
            puan += ödül
        print("Bölüm={}   Puan={}   Durum={}".format(bölüm, puan, durum))


env = make_atari_env('Breakout-v0', n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)
döküm_kayıt_yolu=os.path.join("Eğit","Döküm")
tapçan_kayıt_yolu=os.path.join("Eğit","Tapçan")
def kaydet():
    print("zaten var") if os.path.exists(tapçan_kayıt_yolu) else os.mkdir(tapçan_kayıt_yolu)
    tapçan=A2C("CnnPolicy",env,verbose=1,tensorboard_log=döküm_kayıt_yolu)
    tapçan.learn(total_timesteps=20000)
    tapçan.save(os.path.join(tapçan_kayıt_yolu,"A2C_20.000_.zip"))
    del tapçan
# kaydet()
def yükle():
    tapçan=A2C.load(os.path.join(tapçan_kayıt_yolu,"A2C_20.000_.zip"),env)
    print(tapçan.action_space)
    return tapçan
# yükle()
def değerlendir():
    tapçan=yükle()
    env = make_atari_env('Breakout-v0', n_envs=1, seed=0)
    env = VecFrameStack(env, n_stack=4)
    sonuç=evaluate_policy(tapçan,env,n_eval_episodes=10,render=True)
    env.close()
    print(sonuç)
# değerlendir()
# env = make_atari_env('Breakout-v0', n_envs=4, seed=0)
# env = VecFrameStack(env, n_stack=4)
# log_path = os.path.join("Eğit","Döküm")
# model = A2C("CnnPolicy", env, verbose=1, tensorboard_log=log_path)
# model.learn(total_timesteps=400000)
a2c_path = os.path.join("Eğit","Tapçan","A2C_10.000_.zip")
# model.save(a2c_path)
# del model
env = make_atari_env('Breakout-v0', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)
model = A2C.load(a2c_path, env)
sonuç=evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(sonuç)