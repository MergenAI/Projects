"""
https://pypi.org/project/gym-super-mario-bros/
https://www.youtube.com/watch?v=ZxXKISVkH6Y
"""
import os
# Import PPO for algos
from stable_baselines3 import PPO,A2C
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Import policy for hyper parameters
from stable_baselines3.common.policies import BasePolicy
# Import the game
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy

env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 2. Simplify the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')
state = env.reset()
# state, reward, done, info = env.step([env.action_space.sample()])
# plt.figure(figsize=(10,8))
# for idx in range(state.shape[3]):
#     plt.subplot(1,4,idx+1)
#     plt.imshow(state[0][:,:,idx])
# plt.show()
# plt.imshow(state)
# plt.show()

class TrainAndLogCallback(BaseCallback):

    def __init__(self, check_frequency, save_path, verbose=1):
        super(TrainAndLogCallback, self).__init__(verbose)
        self.check_freq = check_frequency
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
CHECKPOINT_DIR = './train/new/best_model_10000.zip'
LOG_DIR = './logs/new/'
# # Setup model saving callback
# callback = TrainAndLogCallback(check_frequency=10000, save_path=CHECKPOINT_DIR)
# # This is the AI model started
# model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=1e-04,
#             n_steps=128)
# # Train the AI model, this is where the AI model starts to learn
# model.learn(total_timesteps=10000, callback=callback)
# model.save('super-mario')

model=PPO.load(path=CHECKPOINT_DIR,env=env)
print(evaluate_policy(model,env, n_eval_episodes=10, render=False))
for i in range(1,6):
    state=env.reset()
    done=False
    score=0
    while not done:
        env.render()
        hareket,_sec_state=model.predict(state)
        nstate,reward,done,info=env.step(hareket)
        score+=reward
    print("Episode={}   Score={}   State={}".format(i,score,state))


