# PPO：把策略更新限制在可靠范围内

PPO（Proximal Policy Optimization）是策略梯度家族里非常常用的算法。它的核心目标不是提出一个全新的学习信号，而是让策略更新别走得太远。

<img src="../picture/ppo_clip_objective.png" width="700" alt="ppo clip objective">

## 为什么需要 PPO

普通策略梯度用当前策略采样，再用这些样本更新当前策略。问题是策略一更新，数据分布就变了。旧数据来自 $\pi_{\theta_{\text{old}}}$，新策略是 $\pi_\theta$，如果两者差距太大，用旧数据估计新策略梯度就会不稳定。

重要性采样可以把旧策略数据修正到新策略：

$$
\mathbb{E}_{a\sim \pi_\theta}[f(a)]
=
\mathbb{E}_{a\sim \pi_{\theta_{\text{old}}}}
\left[
\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)}f(a)
\right]
$$

令：

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

策略目标可以写成：

$$
L(\theta)=\mathbb{E}[r_t(\theta)A_t]
$$

但如果 $r_t$ 过大或过小，方差会变大，训练会崩。

## PPO-clip

PPO-clip 直接把 ratio 限制在一个小区间：

$$
L^{CLIP}(\theta)=
\mathbb{E}\left[
\min\left(
r_t(\theta)A_t,
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

如果 $A_t>0$，说明这个动作比平均好，我们希望提高它的概率。但提高到 $1+\epsilon$ 之后，就不再给额外收益。

如果 $A_t<0$，说明这个动作不好，我们希望降低它的概率。但降低到 $1-\epsilon$ 之后，也不再鼓励继续远离旧策略。

所以 clip 的作用是：允许策略朝正确方向更新，但不允许一步跨太大。

## PPO-penalty

另一种 PPO 写法是在目标里加入 KL 惩罚：

$$
L^{KL}(\theta)=\mathbb{E}[r_t(\theta)A_t]-\beta KL(\pi_{\theta_{\text{old}}},\pi_\theta)
$$

它和 TRPO 的思想接近：新旧策略行为距离不能太大。PPO-clip 更常见，因为实现简单，不需要显式处理复杂约束。

## PPO 的训练流程

1. 用当前策略 $\pi_{\theta_{\text{old}}}$ 与环境交互，收集一批轨迹。
2. 用价值函数估计 $V(s)$，计算 advantage。
3. 固定旧策略概率，反复优化 clip objective 若干 epoch。
4. 更新旧策略为当前策略，重新采样。

PPO 仍然通常被视为 on-policy 算法，因为它只复用“近端”的旧数据，而不是像 DQN 那样长期使用 replay buffer。

## 小结

PPO 的价值在工程上非常直接：比原始策略梯度更稳定，比 TRPO 更容易实现。它的关键不是 clip 公式本身，而是“策略更新必须受约束”这个思想。

