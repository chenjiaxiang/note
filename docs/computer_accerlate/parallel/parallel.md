# 并行计算
并行计算主要讨论三个部分。  
1. 具体的并行算法：数据并行，模型并行(张量并行，流水线并行)，专家并行等。
2. 通信算法：并行的时候是需要信息交互的，信息如何高效交互由通信算法负责。
3. 计算框架：在工程上结合以上两者的实现。具体有pytorch的DP，DDP, FSDP, ZeRO(Zero Redundancy Optimizer), DeepSpeed, Megatron等。