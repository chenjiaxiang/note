# LAS
LAS（Listen Attend Spell）也称为AED（Attention based Encoder Decoder）。整体结构如下图  
<span style="color: red;">加图</span>   
先根据公式对这两个名称对应的模块做下介绍。公式如下：
$$c_u = \sum_{t} \alpha_{u,t} \cdot h_t, \quad \alpha_{u,t} = \text{softmax}_t(\text{score}(s_{u-1}, h_t))$$
$$P(y \mid x) = \prod_{u=1}^{U} P(y_u \mid y_{1:u-1}, x_{1:T})$$
其中LAS的listen对应AED的encoder部分，也就是图中的音频编码单元，LAS的Spell对应AED的decoder部分，也就是图中的文字解码部分，LAS的Attend对应AED的Attention，对应图中的decoder解码时对encoder编码结果的使用。  
AED其实就是传统的ED结构的ASR模型与Attention的结合。以前的RNN形式的encoder在编码完后输出一个包含所有信息的token，然后decoder使用这个token和前置解码结果来解码一个输出符号。