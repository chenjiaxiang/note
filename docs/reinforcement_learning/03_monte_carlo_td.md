# 蒙特卡洛与时序差分

当环境模型未知时，我们无法枚举 $P(s'|s,a)$ 来做 Bellman 期望备份，只能让智能体与环境交互，拿到轨迹后估计价值函数。蒙特卡洛（Monte Carlo, MC）和时序差分（Temporal Difference, TD）是两种基本估计方式。

<img src="../picture/mc_vs_td.png" width="700" alt="mc vs td">

## 蒙特卡洛方法

MC 的思想非常直接：等一个 episode 结束，计算真实回报 $G_t$，然后用它更新状态价值：

$$
V(s_t)\leftarrow V(s_t)+\alpha\left(G_t-V(s_t)\right)
$$

其中：

$$
G_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots
$$

MC 不需要环境模型，也不需要当前价值函数估计下一步价值。它使用完整轨迹给出的真实回报，所以估计偏差小。

但 MC 的方差通常很大。因为一个状态后面可能经历很长的随机过程，最终回报受很多后续随机事件影响。另一个限制是 MC 通常需要 episode 结束后才能更新，不适合无限长或很长的持续任务。

## 时序差分方法

TD 不等 episode 结束，它走一步就更新一次：

$$
V(s_t)\leftarrow V(s_t)+\alpha\left(r_t+\gamma V(s_{t+1})-V(s_t)\right)
$$

括号里的项叫 TD error：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

TD 的目标不是完整真实回报，而是一步奖励加上下一个状态的估计价值。这种“用估计更新估计”的方式叫 bootstrapping。

TD 的优点是更新及时、方差更低，也能处理持续任务。缺点是它依赖 $V(s_{t+1})$ 的估计，如果当前价值函数很差，更新目标也会有偏差。

## Bias 与 Variance

MC 和 TD 的核心差异可以用 bias-variance 来理解。

MC 使用真实采样回报，偏差低，但完整轨迹中的随机性都会进入目标，方差高。

TD 使用自举目标，方差低，但目标包含当前模型估计，因此有偏差。

强化学习里很多算法都是在这个权衡上移动。例如 $n$-step TD 用未来 $n$ 步真实奖励加上第 $n$ 步后的价值估计：

$$
G_t^{(n)}=r_t+\gamma r_{t+1}+\cdots+\gamma^{n-1}r_{t+n-1}+\gamma^nV(s_{t+n})
$$

$n=1$ 时接近 TD；$n$ 越大越接近 MC。

## 为什么重要

MC 和 TD 是后续控制算法的基础。Sarsa 和 Q-learning 可以看成把 TD 更新从状态价值 $V(s)$ 推广到动作价值 $Q(s,a)$；DQN 又把表格里的 $Q(s,a)$ 换成神经网络。

