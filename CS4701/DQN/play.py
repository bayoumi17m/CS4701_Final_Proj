import matplotlib
matplotlib.use("Agg")
from DQN import DuelingDQNPrioritizedReplay
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
from gym.wrappers import Monitor
import warnings
import os
import time

def play(agent,env):

    load_path = "./tmp/PRmem/model.ckpt"
    agent.load(load_path)

    obs = env.reset()
    score = 0
    steps = 0
    done = False
    index =[]
    val = []

    while done == False:
        steps += 1
        index.append(steps)

        #action = agent.pick_action(obs)
        obs, rewards, done, _ = env.step(env.action_space.sample())
        val.append(rewards)

        score += rewards

        env.render()
        time.sleep(0.1)

if __name__ == "__main__":
    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config.gpu_options.allow_growth = True
    sess = tf.Session(config= config)

    env = gym.make('MsPacman-ram-v0')
    MEMORY_SIZE = 5000
    ACTION_SPACE = 9
    OBSERVATION_SPACE = 128 # 210*160*3
    EPISODES = 1
    STEPS = 45000
    IMAGE_WIDTH = 210
    IMAGE_HEIGHT = 160
    IMAGE_CHANNELS = 3

    with tf.variable_scope('PRmem'):
        prmem_DQN = DuelingDQNPrioritizedReplay(
            n_actions=ACTION_SPACE, n_features=OBSERVATION_SPACE, memory_size=MEMORY_SIZE, epsilon=0,
            epsilon_increment=0, sess=sess, dueling=False, output_graph=True,
            prioritized=True,image_data=False,image_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS))
        print("Prioritized Replay DQN Built")

    sess.run(tf.global_variables_initializer())

    play(prmem_DQN,env)