# 并行训练的通信
在[并行计算](parallel.md)部分，介绍了各种并行算法，以ZeRO1/2/3为例，这几种并行策略都是需要进行通信的，因此这部分介绍通信相关基础。将从通信原语，通信算法两个方面介绍。
## 通信原语
聚合式通信原语（Collective Communication Primitives）的含义是：数据如何在多个rank间分布/聚合。 主要有以下几种：
1. Send/Recv(P2P)： 点对点通信，一个rank给另一个rank发送一段数据。其实只用P2P就可以构建所有的通信原语，但是考虑到效率等问题，还是设计了其他的通信原语。P2P的最常见的通信场景是流水线并行(Pipeline Parallelism)
2. Broadcast: Root rank把一段数据原样发送给所有其他rank，每个rank收到完整的，相同的数据。Broadcast的常见场景是在训练开始时rank0把模型参数广播到所有rank，保证初始权重一致。
3. Scatter: Root Rank将一个大数据块切成N份，分别发送给不同的N个rank。Scatter的常见场景是数据并行（Data Parallel）的数据分发。
4. Gather: Scatter的逆操作，每个rank都发送一份数据给Root rank, root收到完整拼接结果。Gather的常见场景是DP的梯度同步的第一步。
5. AllGather: 每个rank都发送$M/N$的数据，其他rank最后都有完整的结果M。不做具体说明是因为有不同的操作方式可以实现这一目标。AllGather的常见使用场景是ZeRO3中的前向后向时临时收集模型参数的操作。
6. Reduce: 每个rank都提供一份数据，按照指定算子（例如sum或ave）进行聚合。Reduce的名字其实就是相对于纯拼接成一份的拼接操作有额外的聚合操作。  
7. AllReduce: 聚合后的结果让所有rank都有。AllReduce的常见使用场景是DDP的梯度同步，也就是把各个rank的梯度聚合后，再同步给每个rank。
8. ReduceScatter: 每个rank提供一份数据，基于所有rank提供的数据做聚合， 再把聚合结果切分成N份，每个rank保留一份。ReduceScatter的常见使用场景是ZeRO2/3的梯度同步，梯度要先聚合，容纳后每个rank都只保留自己负责维护的数据。
9. AllToAll: 每个rank同事向其他rank发送不同的数据分片，也从所有其他rank接收不同的数据分片。是一个$NxN$的全交换矩阵。AllToAll的常见使用场景是MoE，因为token到每个expert的路由路径是都有可能的。既要把自己的token路由到其他expert，又要从其他expert接收token。但是AllToAll的通信网络连接较多。
10. Barrier: 负责同步的原语，不负责传输数据。但是也很重要，常见的使用场景是存储checkpoint，要保证通信完成了，存储的是正确的checkpoint。

!!! Reduce的补充说明"
用Hadoop处理过数据的话，应该对google的[Map_Reduce](https://static.googleusercontent.com/media/research.google.com/zh-CN//archive/mapreduce-osdi04.pdf)是熟悉的，map-reduce的含义做下解释：map是针对并行的数据进行处理，reduce是对各个并行的结果进行通信聚合。 详细介绍可以看[databricks的说明](https://www.databricks.com/blog/what-is-mapreduce)
---

## 通信算法
通信算法要完成的任务是：在给定了通信原语（也即要完成的通信任务）和物理拓扑（GPU，机器的数量和连接关系）两种条件下，具体怎样实现数据传输。举例：在不考虑效率的情况下，其实所有的通信都用P2P来完成也是可以的。那就可以认为这是一种通信算法。接下来介绍在工程实践中常用的通信算法。以下通信算法在NCCL中基本都有使用，也会介绍不同的数据交换要求下各个通信算法的优劣势以及NCCL是如何自适应选取通信算法的。