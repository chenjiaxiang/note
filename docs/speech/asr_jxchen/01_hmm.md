# HMM的含义以及在语音识别中的应用
HMM全称为Hidden Markov Model，其中markov部分是描述有依赖关系的离散随机过程的，称为markov chain。
## Markov Chain
依赖关系为一个节点只依赖前置节点，依赖的数量称为阶，例如$S_{t+1}$只依赖$s_{t}$，则称为1阶markov  chain。概率描述为：$P(S_{t+1}​=x∣S_t​,S_{t−1}​,…,S_1​)=P(S_{t+1}=s∣S_t​)$，其中$t$是时间步，$S_{t}$是在时刻$t$的随机变量，s是随机变量的取值。

<span style="color: red;">加图</span>
描述一个1阶markov chain，需要的信息是转移概率：在一个状态下转移到另一个状态的概率。公式描述：$p_{ij}​=P(S_{t+1​}=j∣S_t​=i)$，当有n个离散状态时，所有状态之间的转移矩阵如下：
$$
P = \begin{pmatrix}
p_{11} & p_{12} & \cdots & p_{1n} \\
p_{21} & p_{22} & \cdots & p_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
p_{n1} & p_{n2} & \cdots & p_{nn}
\end{pmatrix}
$$
当这个转移概率矩阵的所有值都知道的情况下，这个markov chain就确定了。(转移概率和时间无关的特性叫做时间齐次性。)
随机过程中确定markov chain的方式一般是统计，其中$N_{ij}$表示从状态$i$转移到状态$j$的次数。
$$\hat{p}_{ij} = \frac{N_{ij}}{\sum_{k} N_{ik}} = \frac{N_{ij}}{N_i}$$

## Hidden
接下来讨论hidden的含义，隐指的是刚才描述的$S_i$是无法直接观察的，可以观察的状态用$O_i$表示。$O_i$是$S_i$的条件随机变量。公式表达$b_j(k) = P(O_t = k \mid S_t = j)$，这个概率称为发射概率，发射概率矩阵如下：
$$
B = \begin{pmatrix}
b_1(1) & b_1(2) & \cdots & b_1(M) \\
b_2(1) & b_2(2) & \cdots & b_2(M) \\
\vdots & \vdots  & \ddots & \vdots  \\
b_N(1) & b_N(2) & \cdots & b_N(M)
\end{pmatrix}
$$
与Markov Chain的确定一样，当发射概率矩阵的所有值都知道，发射过程也就确定了。发射概率的确定一般也是基于统计：
$$\hat{b}_j(k) = \frac{\sum_{t=1}^{T} \mathbf{1}[S_t = j,\ O_t = k]}
                    {\sum_{t=1}^{T} \mathbf{1}[S_t = j]}$$

!!! note "为什么会有Hidden Markov Model"
    在实际应用中，我们希望了解的过程中的重要变量很多的确是无法观察的。例如人的情绪，取值定为(高兴，悲伤，愤怒)。我们想要知道一个人的情绪，往往是从行为推断的，行为空间定为(看电影，睡觉，唱歌)。在不同的情绪下发生不同行为的概率不同。当我们对人的情绪感兴趣，需要从行为出发进行推断的时候，其实就是一个HMM，隐变量就是情绪，观测变量就是行为。
---

## HMM的学习过程
上面分别讲了Markov Chain的转移概率矩阵和隐变量的发射概率矩阵的确定过程。
