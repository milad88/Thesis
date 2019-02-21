import gym

#env = gym.make("FetchSlide-v1")
#env.render()

#env = gym.make('Copy-v0')
#env.reset()
#env.render()
import os
import tensorflow
val = '/home/milad/.mujoco/mjpro150/bin'
os.environ['LD_LIBRARY_PATH'] = val

#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import gym
env = gym.make('fetch-v0')
env.reset()
env.render()