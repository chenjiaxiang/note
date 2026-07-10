# Sequence-Level  Knowledge  Distillation[paper](https://arxiv.org/abs/1606.07947)
## 背景
1. 蒸馏的一般目的是模型小型化
2. 蒸馏方式中常见的是蒸表征，蒸输出分布
3. seq2seq任务也用蒸馏
4. seq2seq任务在解码时是前后有依赖的，常用beam search解码
5. 怎么beam search这种考虑更多依赖关系的方式结合到蒸馏中（已有的方式是独立蒸馏token级别）
ps: 基础任务是NMT（neural machine translation）

## 贡献
1. 在token独立蒸馏（word-level）的基础上，提出sequence-level蒸馏
2. 继续提出sequence-level interpolation的方法，把GT作为隐序列（类似于HMM）
3. 分析的角度比较新颖，为提升蒸馏效率提供了一些切入点

## 蒸馏方式解析
一般蒸馏的方式：  
$\mathcal{L}(\theta; \theta_T) = (1 - \alpha) \mathcal{L}_{\text{NLL}}(\theta) + \alpha \mathcal{L}_{\text{KD}}(\theta; \theta_T)$  
第一项是从GT中学习的loss，第二项是蒸馏的loss
### word level
<img src="../../image/distillation/word_level.png" alt="word level" width="800"/> </br>
说明：  
1. 直接蒸馏序列中的每个token的所有概率分布
2. 和GT一起学

### seq level
<img src="../../image/distillation/seq_level.png" alt="word level" width="800"/> </br> 
说明：  
1. 蒸馏的目标不再是每个token对应的所有分布，而是在teacher model上beam search得到的多个序列的概率训练  
2. 和word level的相似性：仅学被beam search搜出来的序列概率  
3. 和GT的相似性：把GT也看成分布（一种退化了的分布， 概率密度集中在一个token上），seq level是调整了概率分布。
4. 为什么这种方式是考虑了全局信息？beam search的结果中就是考虑了全局信息的。在使用word level蒸馏方式时，学习一个token的分布时，可能不在beam search序列中的token会有最大值，但是seq level就不学这个。
5. 也和GT一起学

### seq level interpolation
<img src="../../image/distillation/seq_level_inter.png" alt="word level" width="800"/> </br>
说明：  
1. 上面两种方法都是和GT一起学，seq level interpolation选择不一起学。
2. 把GT和beam search结合起来：beam search搜出来的结果不一定合理。多搜一些，然后和GT算相似度或者距离，选相近的序列进行蒸馏学习。
3. 提出了一个比较有意思的观点：真实序列假装是观察不到的，用seq-level interpolation得到的序列是真实序列加上噪声得到的可观察序列（很像HMM）。也可以看作加噪增强。


## 讨论
1. 论文提出了学习概率分布的效率的概念：直接从GT学的概率分布学到的量是0.9%（平均，自己的理解是每个序列的平均），seq-level的量是16.9%，seq-level的量是7.9%。
2. 对比word level蒸馏和label smoothing: 都是对概率做了调整，word level是通过学习根据数据内部的情况对分布进行调整，label smoothing是基于规则硬调（不考虑数据本身的情况），但是应该对加速训练都有好处。
3. 回顾GT这种退化分布，其实是效率较低的，当有teacher model的时候: 以ASR为例，一段WAV其实可能对应的TEXT不是完全确定的，label只是人为强行打上标签, 当有模型学了大量数据后，模型自身就会调整这个概率，这种方式更加好。
4. 蒸馏扩展：以前的方式主要是大蒸小，fastspeech的蒸馏方式有些启发，teacher模型的结构或者训练方式适合或者能够学到一些特性（不限于模型大小，例如自回归），student模型没有或者对应能力比较弱，那就蒸teacher。