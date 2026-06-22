# MOSHI [paper](https://arxiv.org/html/2410.00037?_immersive_translate_auto_translate=1)
## 背景
传统的对话系统: 一个对话轮次由 ASR + NLP(NLU+NLG) + TTS 三部分组成。
希望实现实时对话系统的困难: 其中ASR和TTS可以通过流式加速, 但是想要借助LLM的能力, LLM的在这种交互模式下的计算速度是实现实时对话的阻碍。

可以尝试的方向:
1. 加速LLM的计算速度。
2. 把语音相关的计算直接嵌入LLM中。

基于第二种方式需要的技术
1. 语音token化, 使其可以接入LLM直接参与计算。[SoundStream](./tokenizer/SoundStream.md) [SpeechTokenizer](./tokenizer/SpeechTokenizer_USLM.md) [compare](./tokenizer/discrete_continous_token_compare.md) [semantic token review](https://ar5iv.labs.arxiv.org/html/2205.10643?_immersive_translate_auto_translate=1)
2. 模型需要同时进行听和说, 要带文本输出时还需要具有写能力。使用不同的流来分别处理听,说,写的部分。
3. 多个流之间需要信息交互。
部分相关论文和开源项目: [GLM-4-Voice](https://github.com/THUDM/GLM-4-Voice) [hertz-dev](https://github.com/Standard-Intelligence/hertz-dev) [mini-onmi](https://github.com/gpt-omni/mini-omni) [LLaMA-Omni](https://github.com/ictnlp/LLaMA-Omni) 
4. 针对实时性的计算加速。

## 模型
### 语音编解码器
<video controls width="1080">
    <source src="https://pub-0041adf75d3e4d7d9a5dce556e21001c.r2.dev/audio-token-video/Neural_Encoder.mp4" type="video/mp4">
    MP4 video is not supported by the web browser.
</video>


1. 相对于SpeechTokenizer, 把卷积替换成了linear和transformer。  
2. 蒸馏语义的模型从Hubert换成了WavLM。  
3. 语义token由一个VQ单独产生, 不由RVQ的第一层输出。  
4. 输入语音和重构语音只做对抗损失(论文报告是因为这样的损失设置得到的合成效果更好)。  

### VQ和RVQ
<video controls width="1080">
    <source src="https://pub-0041adf75d3e4d7d9a5dce556e21001c.r2.dev/audio-token-video/VQ.mp4" type="video/mp4">
    MP4 video is not supported by the web browser.
</video>
<video controls width="1080">
    <source src="https://pub-0041adf75d3e4d7d9a5dce556e21001c.r2.dev/audio-token-video/RVQ.mp4" type="video/mp4">
    MP4 video is not supported by the web browser.
</video>

### RQ-transformer
<video controls width="1080">
    <source src="https://pub-0041adf75d3e4d7d9a5dce556e21001c.r2.dev/audio-token-video/RQ_transformer.mp4" type="video/mp4">
    MP4 video is not supported by the web browser.
</video>


1. 其实这里是提速的一种方式: 如果让Temporal Transformer来产生所有流需要的token, 每个时刻需要产生17个token。为了保证LLM的拟合能力, 需要保障参数与计算量。视频中的生成方式Temporal Transformer每时刻只用生成1个token, 然后基于这个token用规模更小的Depth Transformer来生成17个token, 可以提升生成速度, 对实时性有帮助。 

### headline
<video controls width="1080">
    <source src="https://pub-0041adf75d3e4d7d9a5dce556e21001c.r2.dev/audio-token-video/HeadLine.mp4" type="video/mp4">
    MP4 video is not supported by the web browser.
</video>


1. 示例了acoustic token和semantic token在时间维度上的相对偏移。  
2. 示例了各个流是怎么在RQ-Transformer中进行计算的。  
3. 示例了多个流之间的信息是通过加和与depth Transformer来实现交互的。(讨论点: 在Depth Transformer维度上, moshi的流更加靠前, 感觉不是很合理, 直观上应该把user的流放在更靠前)

## 数据与训练
### 数据
1. 文本数据: 12.5%的高质量数据和87.5%的过滤后的CommonCrawl数据。
2. 语音数据: 
- 7 million hour的语音数据, 用whisper转写出text。语音数据为24KHz单声道, 用来训练单流模型(是语音编解码模块吗？)。
- multi-stream数据为Fisher dataset, 2000 hour的电话对话语音数据, 由随机成对参与者在给定的topic上聊天形成。原始数据为8KHz, 上采样到24KHz。 
- supervised multi-stream dataset。由170 hour带有文字的语音双人语音对话组成。数据质量较高, 可以用来调优基于Fisher训练的模型。(但是论文说moshi训练中没有直接使用这个数据, 而是用它训练了一个多流TTS模型)。对话文本还用来调优了Helium LLM。
3. 语音-文本指令数据: 由人和Helium进行文本对话产生文本, 由多流的流式TTS合成语音。超过20 hours。其中用于训练moshi流的语音数据由单一演员录制(确保了音色一致), 超过70种风格(使moshi的语音表现力更强)。用户流合成音色是随机选的(使模型对各种各样的用户语音建模能力更鲁棒)。


### 训练  
1. Helium LLM: 基于[open-Hermes](https://huggingface.co/datasets/teknium/OpenHermes-2.5/blob/main/README.md)训练, 对话文本调优。  
2. Moshi Pre-Training: Temporal Transformer用Helium初始化, Depth Transformer随机初始化。使用单流方式训练, 文本30%进行mask, 对齐的文本和语音会随机在 ±0.6s之间延迟, 50%的时间训练Helium的训练数据(防止遗忘)。特点: 每个batch塞了超过5 million个序列, 大概对应16个小时的数据。训练数据来源为语音数据的第一部分。  
3. Moshi Post-Training: 基于多流数据训练多流建模能力, 每个batch对应8小时左右语音数据。训练数据来源为经过说话人分离的语音数据的第一部分。文本与语音之间不设置延迟。  
4. Moshi Finetune: (论文认为模型到目前为止处理自然对话的能力还不足, 比如语音重叠和用户长时间静默)使用Fisher dataset进行finetune。最后使用指令数据训练, 固化moshi的音色和人设(对用户流数据做了一些data augmention)。
多流TTS training: 基于多流数据训练。

loss:   
                $L(V, l) = \frac{1}{S} \sum_{s=1}^S \left(CE(u_{s,1}, V_{s,1}) + \frac{1}{\sum_{k=2}^K \alpha_k} \sum_{k=2}^K \alpha_k CE(u_{s,k}, V_{s,k})\right)$

权重参数: semantic对应为100, acoustic对应为1

## 实验结果
1. LLM能力

| | ARCe | ARCc | OBQA | HS | WG | PIQA | SIQA | TQA | NQ | MMLU |
|-|------|------|------|----|----|------|------|-----|----|------|
| Helium | <span style="color: green;">79.6</span> | <span style="color: green;">55.9</span> | 53.6 | 76.3 | <span style="color: green;">70.0</span> | 79.4 | <span style="color: green;">51.0</span> | <span style="color: green;">59.9/72.6</span> | 23.3 | <span style="color: green;">54.3</span> |
| MPT Falcon | 70.5 | 46.5 | 51.4 | <span style="color: green;">77.6</span> | 69.9 | <span style="color: green;">80.6</span> | 48.5 | -/61.2 | 20.8 | 30.8 |
| Llama 2 | 75.2 | 45.9 | <span style="color: green;">58.6</span> | 77.2 | 69.2 | 78.8 | 48.3 | -/72.1 | <span style="color: green;">25.7</span> | 45.3 |
| OLMo | 67.2 | 42.5 | 50.0 | 75.5 | 69.8 | 77.5 | - | -/- | - | 52.0 |
| Mistral | 80.5 | 54.9 | 52.2 | 81.0 | 74.2 | 82.2 | 47.0\* | 62.5/- | 23.2 | 62.5 |
| Gemma 1 | 81.5 | 53.2 | 52.8 | 81.2 | 72.3 | 81.2 | 51.8 | 63.4/- | 23.0 | 64.3 |

- 后面两个模型的规模与训练量都较大, 效果不如。不参与比较。  
- 在前面的几个LLM的比较中, Helium在较多benchmark上都占优。  

2. 语音质量

| Model | fs | fr | bitrate | causal | ABX :arrow_down: | VisQOL :arrow_up: | MOSNet :arrow_up: | MUSHRA(subjective) :arrow_up: |
|-------|---|---|---------|--------|---------|------------|------------|------------|
| Ground Truth | 24kHz | - | - | - | - | - | 3.08 | 90.6±1.0 |
| RVQGAN | 24kHz | 75Hz | 1.5kbps | - | 1.74 | 2.74 | 31.3±1.3 |
| SemanitCodec | 16kHz | 50Hz | 1.3kbps | | 42.2% | 2.43 | <span style="color: green;">3.12</span> | 64.8±1.5 |
| SpeechTokenizer | 16kHz | 50Hz | 1.5kbps | | <span style="color: green;">3.3%</span> | 1.53 | 2.67 | 45.1±1.5 |
| SpeechTokenizer | 16kHz | 50Hz | 4.0kbps | | <span style="color: green;">3.3%</span> | <span style="color: green;">3.07</span> | 3.10 | 74.3±1.5 |
| Mimi, adv. loss only | 24kHz | 12.5Hz | 1.1kbps | ✓ | 8.7% | 1.84 | 3.10 | <span style="color: green;">81.0±1.3</span> |
| Same, downsampled at 16kHz | 16kHz | 12.5Hz | 1.1kbps | ✓ | - | - | - | 77.7±1.4 |
| Mimi, non adv. only | 24kHz | 12.5Hz | 1.1kbps | ✓ | 8.1% | 2.82 | 2.89 | 58.8±1.8 |

- SpeechTokenizer的效果不错, MIMI没有一致优于它。(不直接使用是因为音频的采样率吗？)
- 这里说明了只使用对抗损失会提升生成的语音质量。

3. 实时性
    暂无
4. 问答质量
| Model | Web Q. | LlaMa Q. | Audio Trivia QA |
|-------|---------|-----------|-----------------|
| **Audio only** |  |  |  |
| GSLM (Lakhotia et al., 2021) | 1.5 | 4.0 | - |
| AudioLM (Borsos et al., 2022) | 2.3 | 7.0 | - |
| TWIST (7B) (Hassid et al., 2023) | 1.1 | 0.5 | - |
| Moshi (w/o Inner Monologue) | <span style="color: green;">9.2</span> | <span style="color: green;">21.0</span> | 7.3 |
| **Text and audio** |  |  |  |
| SpeechGPT (7B) (Zhang et al., 2024a) | 6.5 | 21.6 | 14.8 |
| Spectron (1B) (Nachmani et al., 2024) | 6.1 | 22.9 | - |
| Moshi | <span style="color: green;">26.6</span> | <span style="color: green;">62.3</span> | <span style="color: green;">22.8</span> |
| Moshi (w/o text batches in pre-training) | 23.2 | 61.3 | 18.3 |
| **Text** |  |  |  |
| Helium (text) | 32.3 | 75.0 | 56.4 |

- 准确度指标遥遥领先。感觉有点好过头，baseline选择是不是太弱了？而且这里使用准确度感觉不太合理, 采用一些对回答的主观评测会不会更合理？

## 讨论
1. 在多流建模时, 是否有必要对用户流进行建模？
2. 如果建模, Depth Transformer的建模顺序是否合理？
