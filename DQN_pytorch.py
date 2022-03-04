from turtle import forward
from xml.etree.ElementTree import tostring
import gym
import random
import torch
import torch.nn as TNN
import torch.nn.functional as TF
import numpy as np
from collections import deque
import torch.utils.data as Data
import matplotlib.pyplot as plt
ENV_NAME = 'CartPole-v1'
BUFFER_SIZE = 100000
INITIAL_EPSILON=0.5
FINAL_EPSILON=0.01
GAMMA = 0.9
BATCHSIZE = 32

TEST = 10
SAVINGPATH = "./model1/"
#Network
class DQN(TNN.Module):
    def __init__(self,env) :
        super(DQN,self).__init__()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n 
        self.linear_relu_stack = TNN.Sequential(
            TNN.Linear(self.state_dim,20),
            TNN.ReLU(),
            TNN.Linear(20,self.action_dim)
        )
        self.replaybuffer = deque(maxlen=BUFFER_SIZE)
        self.ite=0
        self.epsilon = INITIAL_EPSILON
        self.train_setup()

    def forward(self,x):
        Q_value = self.linear_relu_stack(x)
        return Q_value

    def action(self,state):
        return np.argmax(self.linear_relu_stack(torch.tensor(state)).detach().numpy())

    def egreedy_action(self,state):
        #self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000
        if(random.random()<self.epsilon): return random.randint(0,self.action_dim-1) 
        else: return np.argmax(self.linear_relu_stack(torch.tensor(state)).detach().numpy())
        
    def load_model(self,model_dict):
        self.linear_relu_stack.load_state_dict(model_dict)

    def form_buffer(self,state,action,reward,done,next_state):
        # 独热编码action
        onehot_action = np.zeros(self.action_dim)#action_dim=2
        onehot_action[action]=1
        self.replaybuffer.append([state,onehot_action,reward,done,torch.tensor(next_state)])

        if(len(self.replaybuffer)>BATCHSIZE) :
            self.train_DQN()
    # 训练初始化
    def train_setup(self):
        self.optimizer = torch.optim.Adam(self.linear_relu_stack.parameters(),lr=0.0001)
        self.loss_fn = TNN.MSELoss()

    def train_DQN(self):
        self.ite+=1
        #从buffer中取minibatch
        minibatch = random.sample(self.replaybuffer,BATCHSIZE)
        
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[4] for data in minibatch]

        tensor_state = torch.Tensor(state_batch).reshape(BATCHSIZE,self.state_dim)
        tensor_action = torch.Tensor(action_batch).reshape(BATCHSIZE,self.action_dim)
        
        #由于每次训练网络都会改变
        #所以重新计算y
        y_batch=[]
        for i in range(BATCHSIZE):
            if minibatch[i][3] : 
                y_batch.append(reward_batch[i])
            else : 
                y_batch.append(reward_batch[i] + GAMMA*torch.max(self.linear_relu_stack(next_state_batch[i])).detach().numpy()) 
        tensor_y = torch.Tensor(y_batch)

        pred = self.linear_relu_stack(tensor_state)  #计算DQN对本step的Q_value估计
        Q_action = torch.sum(pred*tensor_action,dim=1)#保留本step采取action对应的Qvalue（DQN预测的）
        loss = self.loss_fn(tensor_y,Q_action)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if (self.ite % 1000) == 0:
            loss, current = loss.item(), self.ite * len(state_batch)
            print(f"loss: {loss:>7f} ite: {self.ite}")

        if (self.ite % 5000)==0:
            fileName = 'model-'+str(self.ite)+'.pth'
            torch.save(self.linear_relu_stack.state_dict(), SAVINGPATH + fileName)
            print(f"============================model ite: {self.ite} ================================")

def MountainCar_agent_reward(next_state,done,i):
    if not done:
        reward_agent =  np.square(abs(next_state[1]))/2 + np.sin(3*next_state[0])*0.0025
    elif i  == 199:
        reward_agent =  np.square(abs(next_state[1]))/2 + np.sin(3*next_state[0])*0.0025
    else:
        reward_agent= -1
    return reward_agent

def CartPole_agent_reward(next_state,done,i):
    if not done:
        reward_agent =  0.2-abs(next_state[0])*0.1#0.1#0.3-np.abs(next_state[0])*0.1
    elif i  == 999:
        reward_agent =  0.2-abs(next_state[0])*0.1
    else:
        reward_agent= -1
    return reward_agent

def main():
    env = gym.make(ENV_NAME)
    dqn = DQN(env)
    episode=0
    Test_rec = []
    while dqn.ite<200000:
        episode+=1
        #i=0
        state = env.reset()
        for i in range(1,1000):
            action = dqn.egreedy_action(state)
            next_state,reward,done,_ = env.step(action=action)

            reward_agent = CartPole_agent_reward(next_state,done,i)
            #对最大奖励取对数 / 无效，直接不收敛，破坏了模型的马尔科夫性，向DQN网络输入step数可能使该操作有效
            #reward_agent = reward_agent*math.log(i)
            if i==999:
                break
            dqn.form_buffer(state,action,reward_agent,done,next_state)
            state = next_state#维护下一轮状态
            if done: #对于环境最大step，直接放弃该数据
                break
        if (dqn.ite%1)==0:
            total_reward=0
            max_a_reward=0
            min_a_reward=1000
            for j in range(TEST): #test 10 
                state = env.reset()
                acc_reward = 0
                for i in range(1000):
                    action = dqn.action(state)
                    state,reward,done,_ = env.step(action=action)
                    acc_reward+=reward
                    if done :
                        break
                    #env.render()
                total_reward += acc_reward
                max_a_reward = max(max_a_reward,acc_reward)
                min_a_reward = min(min_a_reward,acc_reward)
            Test_rec.append([min_a_reward,total_reward/TEST,max_a_reward])
            print(f'acc_reward: {acc_reward}   max_reward: {max_a_reward}   mean_reward: {Test_rec[-1]}')

            if (total_reward/TEST)==1000:
                fileName = 'optim-model-episode'+str(episode)+'.pth'
                torch.save(dqn.linear_relu_stack.state_dict(), SAVINGPATH + fileName)
    while True:
        pass
if __name__ == '__main__':
    main()



#[[1, 9.6], [2, 13.0], [3, 20.2], [4, 11.7], [5, 23.9], [6, 41.4], [7, 23.1], [8, 37.8], [9, 21.8], [10, 29.3], [11, 49.0], [12, 42.9], [13, 47.8], [14, 42.3], ...]