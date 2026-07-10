# SPEECHTOKENIZER: UNIFIED SPEECH TOKENIZER FOR SPEECH LANGUAGE MODELS [paper](https://arxiv.org/html/2308.16692?_immersive_translate_auto_translate=1)
## 目标
1. 探讨哪种针对语音的编码方式更加适合语音与LLM结合

|token种类        |内容准确性           |语音的质量           |单个tokenizer       |
|-----------------|--------------------|--------------------|--------------------|
|Semantic LM      | :heavy_check_mark: | :x:                | :heavy_check_mark: |
|Acoustic LM      | :x:                | :heavy_check_mark: | :heavy_check_mark: |
|Hierarchical LM  | :heavy_check_mark: | :heavy_check_mark: | :x:                |
|USLM(paper)      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

Hierarchical LM的一些典型方法:
[AudioLM](https://ar5iv.labs.arxiv.org/html/2209.03143?_immersive_translate_auto_translate=1)
[AudioPaLm](https://arxiv.org/html/2306.12925?_immersive_translate_auto_translate=1)
[PolyVoice](https://arxiv.org/html/2306.02982?_immersive_translate_auto_translate=1)

论文提出semantic token和acoustic token之间有冗余，会有不必要的额外建模复杂度

## contributions:

1. 提出了SpeechTokenizer。针对Speech Large Language Model设计，统一了semantic和acoustic表征（在RVQ中解耦,论文的统一指的是把两者联合起来）。
2. 建立了SLMTokBench, 专门用来评测speech token对SLM（speech language model）的适配性。
3. 基于SpeechTokenizer建立了USLM（Unified Speech Language Model），并在zero-shot的TTS任务上优于VALL-E。

## SLMTOKBENCH

适合SLM的speech token应该具有的特性：
1. 与文本可以强对齐
2. 保留了丰富的语音信息

### text alignment evaluation
方式：使用表征训练一个语音识别模型(2层1024D的BiLSTM + 字符级别的CTC loss), 然后基于如下loss计算speech token与text的对齐程度。

$\hat{I}(x; y) = \frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N [\log q_0(y_i|x_i) - \log q_0(y_j|x_i)]$

其中x是离散语音表征, y是对应的文本。  

### information preservation evaluation
基于LibriSpeech训练unit-HiFIGAN，实现把Hubert特征转换成语音。
1. 内容：用ASR再对unit-HiFIGAN合成的wav进行识别, 使用ASR衡量对内容的保留程度。
2. 音色：使用WavLM-TDNN来评测decoder产生的语音和gt语音的语者相似度。（在LibraSpeech中选择了300个samples）


## 模型
### SpeechTokenizer
<video controls width="1080">
    <source src="https://pub-0041adf75d3e4d7d9a5dce556e21001c.r2.dev/audio-token-video/Tokenizer_EnDec.mp4" type="video/mp4">
    MP4 video is not supported by the web browser.
</video>


1. 把High  Fidelity  Neural  Audio  Compression论文中的[encoder](./HiFI_Fidelity_Neural_Audio_Compression.md)的双层LSTM换成双层BiLSTM，增强语义建模能力

training loss:  
1. hubert蒸馏loss(实验中最后采用的是哪种需要确认下,用的应该是第一个,因为蒸馏teacher使用的是hubert的L9输出,unit输出应该是对应第二种蒸馏方式)  
    1. continuous distill:  
            $L_{\text{distill}} = -\frac{1}{D}\sum_{d=1}^D \log \sigma(\cos(\text{AQ}_1^{(:, d)}, S^{(:, d)}))$

    2. pseudo-label distill:  
            $L_{\text{distill}} = -\frac{1}{T}\sum_{t=1}^T u^t \log(\text{Softmax}(Aq_1^t))$

2. 重构loss  
    1. 时域波形loss:   
            $L_t = \|x - \hat{x}\|_1$
    2. 频域loss:  
            $L_f = \sum_i \|S_i(x) - S_i(\hat{x})\|_1 + \|S_i(x) - S_i(\hat{x})\|_2$
3. 对抗loss  
    1. 标准对抗loss:  
            $L_D = \frac{1}{K}\sum_{k=1}^K \max(1 - D_k(x), 0) + \max(1 + D_k(\hat{x}), 0)$
    2. feature loss:  
            $L_{\text{feat}} = \frac{1}{KL}\sum_{k=1}^K\sum_{l=1}^L \frac{\|D_k^l(x) - D_k^l(\hat{x})\|_1}{\text{mean}(\|D_k^l(x)\|_1)}$

总loss:  
            $L_G = \lambda_t L_t + \lambda_f L_f + \lambda_g L_g + \lambda_{\text{feat}} L_{\text{feat}} + \lambda_w L_w + \lambda_{\text{distill}} L_{\text{distill}}$

### USLM

<video controls width="1080">
    <source src="https://pub-0041adf75d3e4d7d9a5dce556e21001c.r2.dev/audio-token-video/USLM.mp4" type="video/mp4">
    MP4 video is not supported by the web browser.
</video>


sementic token的产生和loss:  $L_{AR} = -\log \prod_{t=0}^T p(c_1^t|c_1^{<t}, u; \theta_{AR})$

acoustic token的产生和loss(在同一个time step上):  $L_{NAR} = -\log \prod_{i=2}^8 p(c_i|c_{<i}, \hat{C}, u; \theta_{NAR})$


TTS实验中, AR model和NAR model都是12层Transformer解码器, 16个attention head, 1024的attention维度, 4096的FFN输出维度。


## 数据与训练
1. SpeechTokenizer training:  从LibriSpeech的数据中采样3s的片段用来训练。
2. USLM traininig: 多语言的LibriSpeech dataset中选取3~14s的数据作为训练数据, speech数据的采样率都是16KHz。

## 实验结果
### SpeechTokenizer评测
评测标准：
VISQOL(客观评测指标) [old paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/39979.pdf) [code](https://github.com/google/visqol)  
MUSHRA(主观评测指标) [paper](https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1534-3-201510-I!!PDF-E.pdf)

Tokenizer evaluation

|Tokeniz                 |WER(objective):arrow_down:|VISQOL(objective):arrow_up:|MUSHRA(subjective):arrow_up:|
|------------------------|--------------------------|---------------------------|-------------------------|
|GroudTruth              |4.58                      |-                          |91.46                    |
|[EnCodec](./HiFI_Fidelity_Neural_Audio_Compression.md)        |5.11                      |4.37                       |79.86                    |
|SpeechTokenizer         |<span style="color: green;">5.04</span>|<span style="color: green;">4.30</span>|<span style="color: green;">90.55</span>|

### USLM评测  
客观评测指标
1. 生成语音和prompt语音的说话人相似度(使用WavLM-TDNN分别计算两个语音的speaker embedding, 然后计算normalized embedding的余弦相似度)。
2. WER: 使用whisper(medium)模型对合成语音进行ASR, 基于ASR结果计算WER。

主观评测指标
1. MOS(Mean Opinion Score): 12个母语者评测  
2. SMOS(Similarity Mean Opinion Score): 6个母语者评测  

|Model|Tokenizer|WER(objective):arrow_down:|SIM(objective):arrow_up:|MOS(subjective):arrow_up:|SMOS(subjective):arrow_up:|
|-|-|-|-|-|-|
|GroundTruth|-|1.9|0.93|4.5|3.96|
|VALL-E|EnCodec|7.9|0.75|3.08|3.31|
|USLM|SpeechTokenizer|<span style="color: green;">6.5</span>|<span style="color: green;">0.84</span>|<span style="color: green;">3.63</span>|<span style="color: green;">3.45</span>|

SLMTokBench 


| Tokenizer |     | Teacher | MI(TA):arrow_up:| WER(TA):arrow_down:| WER(IP):arrow_down:| SIM(IP):arrow_down:|
| --- | --- | --- | ---     | ---    | ---     | ---     |
| Groundtruth | - | - | - | - | 4.58 | 1.0 |
| HuBERT | KM500 | - | 31.2 | 9.88 | 16.26 |0.77|
| EnCodec | RVQ-1 | - | 16.5 |61.52| 38.34 |0.92|
| EnCodec | RVQ-1:8 | - | 23.6 |30.91| 5.11 |0.98|
| Ablations | | | | | | |
| SpeechTokenizer | RVQ-1 | HuBERT avg | 30.9 |15.58| 9.57 |0.74|
| SpeechTokenizer | RVQ-1:8 | HuBERT avg | 29.7 |16.03| 5.04 |0.97|
| SpeechTokenizer | RVQ-1 | HuBERT L9 | <span style="color: green;">32.9</span> |<span style="color: green;">12.68</span>| 14.17 |0.73|
| SpeechTokenizer | RVQ-1:8 | HuBERT L9 | 31.6 |13.12| <span style="color: green;">5.31</span> |<span style="color: green;">0.97</span>|
| SpeechTokenizer | RVQ-1 | HuBERT units | 24.2 |34.13| 20.02 |0.72|
| SpeechTokenizer | RVQ-1:8 | HuBERT units | 25.1 |30.71| 5.84 |0.95|

(TA表示属于衡量Text Alignment的指标,IP表示属于衡量Information Preservation的指标)
1. 论文描述RVQ-1的SIM很低，RVQ1:8很高，表示音色信息在RVQ-1中去除得较好。
2. Hubert L9作为蒸馏的teacher一致优于Hubert unit, 表示hubert的连续表征比离散的unit保留了更多的content信息
3. 论文认为L9包含更纯净的content信息，ave包含了一些timbre信息

RVQ的解耦表现  
1. VC(voice conversion):

|source|reference|WER:arrow_down:|SIM:arrow_up:|
|-|-|-|-|
|GT|GT|0.4|0.93|
|RVQ-1|RVQ-2|2.6|0.72|
|RVQ-1|RVQ-2:4|11.7|0.80|
|RVQ-1|RVQ-2:8|35.4|0.82|

- RVQ-2:8中, 注入的信息越少, WER越低, 说明RVQ-1主要负责semantic信息, 并且只使用RVQ-1, 已经可以做到很低的WER。
- RVQ-2:8中, 注入的信息越多, 说话人相似度越高, 说明RVQ-2:8主要负责富语言信息, 但是只使用RVQ-1也达到了0.72的说话人相似度。

2. 数据分布
<img src="../../image/DISENTANGLEMENT.png" alt="disentanglement" width="200"/>
(不同说话人选10句话, 经过SpeechTokenizer后在时间维度做pooling, 然后使用t-SNE做降维和可视化)

- RVQ-1在这个分布中是完全分散的，说明其中的与特定说话人的相关信息较少。 
- RVQ-2:8出现明显的聚类现象, 包含丰富的说话人信息。

## 讨论
1. hubert L9 feature作为semantic token怎么就做到了解耦？hubert特征还可以恢复出说话人相似度，应该不属于semantic范围。
2. 评测语音信息时只针对音色进行了评测。情感，音调等信息是否可以进行评测？（根据下游任务，评估是否有评测情感, 音调等富语言信息的必要）
3. RVQ的表征是否有必要解耦得很干净, 如果确实有必要, 那是否可以引入基于人物的判别对抗损失？
