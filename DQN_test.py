import DQN_pytorch
import gym
import torch
# Hyper Parameters for DQN
GAMMA = 0.95 # discount factor for target Q 
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
# Hyper Parameters for game
ENV_NAME = 'CartPole-v1'
EPISODE = 10000 # Episode limitation
STEP = 1000 # Step limitation in an episode
TEST = 100 # The number of experiment test every 100 episode


import os


def main():
    env = gym.make(ENV_NAME)
    dqn = DQN_pytorch.DQN(env)
    
    state = env.reset()
    episode=0
    Test_rec = []
    files = os.listdir("./model/")
    for state_dict in files:
        print("-------------------------------------------------------------")
        print(state_dict)
        dqn.load_model(torch.load("./model/"+state_dict))
        dict_result = []
        total_reward=0
        max_a_reward=0
        for i in range(TEST):
            #for j in range(TEST): #test 10 
            state = env.reset()
            acc_reward = 0
            for i in range(1000):
                action = dqn.action(state)
                state,reward,done,_ = env.step(action=action)
                acc_reward+=reward
                if done :
                    break
                #env.render()
            dict_result.append(acc_reward)
            total_reward += acc_reward
            max_a_reward = max(max_a_reward,acc_reward)
        Test_rec.append([episode,total_reward/TEST,dict_result])
        print(f'acc_reward: {acc_reward}   max_reward: {max_a_reward}   mean_reward: {Test_rec[-1][1]}')
        print(dict_result)


if __name__ == '__main__':
    main()