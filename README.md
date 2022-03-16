# DQN_Pytorch
DQN coded with pytorch

## DQN in Cartpole-v1
模型在训练过程中发生退化的现象，这可能与本项目采用的神经元节点数量有关。value-base的模型存在value function的细微变动却导致策略所产生动作概率大幅改变的情况，这种情形在训练过程中会导致采样轨迹的偏移，从而导致模型的退化，增加网络神经元数量可以提高模型对信息的容纳能力，减小极端数据对value function的不良影响，改善模型表现情况。

此外，单DQN网络对Q函数的估计也会由于噪声的存在而偏高，Nature版本DQN依靠双QNet及延时更新的技术对此现象起到了改善的作用。

## Train and test
可利用的模型参数文件存储在modelDQN文件夹下，DQN_Pytorch.py文件可用于模型的训练
DQN_Test.py用于模型推理
