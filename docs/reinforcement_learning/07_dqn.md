# DQN：用神经网络近似 Q 表

Q-learning 用表格存储 $Q(s,a)$。当状态是图像、连续向量或高维传感器输入时，表格无法覆盖所有状态。DQN（Deep Q-Network）用神经网络近似动作价值函数，把 Q-learning 推到高维状态空间。

<img src="../picture/dqn_training_loop.png" width="700" alt="dqn training loop">

## 从 Q 表到 Q 网络

离散动作场景下，DQN 通常输入状态 $s$，输出每个动作的 Q 值：

$$
Q_\theta(s)=[Q_\theta(s,a_1),Q_\theta(s,a_2),\cdots,Q_\theta(s,a_n)]
$$

执行时选择：

$$
a=\arg\max_a Q_\theta(s,a)
$$

训练目标来自 Q-learning：

$$
y=r+\gamma \max_{a'}Q(s',a')
$$

神经网络用均方误差拟合：

$$
\mathcal{L}(\theta)=\left(Q_\theta(s,a)-y\right)^2
$$

## 经验回放

强化学习数据相邻样本高度相关，直接按时间顺序训练神经网络容易不稳定。DQN 把交互得到的 transition 存进 replay buffer：

$$
(s,a,r,s',done)
$$

训练时从 buffer 随机采样 mini-batch。这样可以打散时间相关性，并让旧经验被重复利用，提高样本效率。

## 目标网络

如果目标 $y$ 也由当前网络 $Q_\theta$ 计算，那么模型一边追目标，一边移动目标，容易震荡。DQN 引入目标网络 $Q_{\theta^-}$：

$$
y=r+\gamma \max_{a'}Q_{\theta^-}(s',a')
$$

目标网络参数每隔若干步从在线网络复制一次，或者用软更新缓慢跟随。这样目标更稳定。

## 常见改进

Double DQN 解决最大化带来的过估计。它用在线网络选动作，用目标网络评估动作：

$$
y=r+\gamma Q_{\theta^-}(s',\arg\max_{a'}Q_\theta(s',a'))
$$

Dueling DQN 把 Q 拆成状态价值和动作优势：

$$
Q(s,a)=V(s)+A(s,a)
$$

这样网络能单独学习“这个状态好不好”和“这个动作相对其他动作好不好”。

Prioritized Replay 不是均匀采样经验，而是优先采样 TD error 大的样本，因为这些样本当前学得不好，更新价值更高。

Rainbow 把多种 DQN 技巧组合在一起，包括 Double、Dueling、Prioritized Replay、多步回报、分布式价值和 NoisyNet。

## 局限

DQN 适合离散动作空间。连续动作下，$\max_a Q(s,a)$ 需要在无限动作空间里优化，直接枚举不可行。DDPG 等 actor-critic 方法就是为连续动作设计的。

