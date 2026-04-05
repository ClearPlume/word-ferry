# 衔言渡意 / Word Ferry

跨语言翻译：将一种语言的意义摆渡到另一种语言。

基于 Transformer 编码器-解码器架构的多语言神经机器翻译系统，支持中文、英文、法文三种语言共六个翻译方向的互译。

## 效果展示

最佳 BLEU 分数：**34.16**

```
输入> 在此框架内，突尼斯各部委、机构和其他有关单位举行了一系列协调会议，以确定执行这一任务的最佳方法。
目标语言(zh/en/fr)> fr
<fr>Dans ce cadre, les ministères, institutions et autres unités concernées de la Tunisie ont organisé une série de réunions de coordination afin de recenser les meilleures méthodes de mise en œuvre de ce mandat.

输入> Espagne
目标语言(zh/en/fr)> zh
<zh>西班牙

输入> The Russian Federation participates in programmes for the construction and operation of ISS and space systems for environmental monitoring, early warning of destructive natural phenomena and other emergency situations, search and rescue operations and programmes to control and reduce pollution in outer space.
目标语言(zh/en/fr)> zh
<zh>俄罗斯联邦参与旨在监测和减少外层空间污染的空间设施和空间系统的建造和运行方案，以便监测环境、预警破坏性自然现象和其他紧急情况、搜索和救援业务和方案。
```

## 模型架构

| 参数             | 值                              |
|----------------|--------------------------------|
| 架构             | Transformer Encoder-Decoder    |
| 参数量            | 51.7M                          |
| 词表大小           | 48,000 (SentencePiece Unigram) |
| 嵌入维度 (d_model) | 384                            |
| 注意力头数          | 6                              |
| 编码器层数          | 8                              |
| 解码器层数          | 8                              |
| FFN 倍率         | 4                              |
| 最大序列长度         | 256                            |

## 特性

- **KV Cache 推理加速** — 自定义解码器实现，推理时缓存 Key/Value，避免重复计算
- **动态 Dropout 调度** — 基于训练/验证损失的统计检验，自动调整 Dropout 率防止过拟合
- **长度分组采样** — 按序列长度分组构建 batch，减少 padding 开销
- **自适应训练控制** — Early Stopping、梯度裁剪、ReduceLROnPlateau 学习率调度
- **权重绑定** — 嵌入层与输出投影层共享权重，减少参数量
- **BLEU 评估** — 训练过程中使用 sacrebleu 在验证集上计算 BLEU 分数
- **TensorBoard 监控** — 记录损失、梯度范数、超参数变化

## 数据

训练数据来源于联合国平行语料库：

> https://www.un.org/dgacm/zh/content/uncorpus/download

选用文件类型：Plain-text bitexts，选用语言：中英法。

每个翻译方向采样 300,000 对，共 6 个方向（en↔zh、en↔fr、fr↔zh），合计 180 万训练样本。

数据处理流程：

1. `scripts/corpus_extractor.py` — 从联合国原始语料文件中解析并抽样翻译对
2. `scripts/sample_generator.py` — 合并各方向语料，添加目标语言标记
3. `scripts/build_vocabulary.py` — 训练 SentencePiece 分词模型（48K 词表）

辅助脚本：

- `scripts/corpus_analyze.py` — 语料统计分析
- `scripts/d_model_calculator.py` — 根据数据规模反推嵌入维度

## 项目结构

```
word-ferry/
├── src/word_ferry/
│   ├── main.py                     # 训练入口
│   ├── trainer.py                  # 训练循环
│   ├── tester.py                   # 交互式推理
│   ├── path.py                     # 路径管理
│   ├── components/
│   │   ├── config.py               # 训练配置
│   │   ├── model.py                # Transformer 模型
│   │   ├── tokenizer.py            # SentencePiece 封装
│   │   ├── logger.py               # 日志
│   │   ├── train/
│   │   │   ├── dataset.py          # 数据集与采样
│   │   │   ├── collate.py          # 批次整理与填充
│   │   │   ├── sampler.py          # 长度分组采样器
│   │   │   └── dropout_scheduler.py # 动态 Dropout 调度
│   │   └── infer/
│   │       ├── cached_decoder.py           # 带缓存的解码器
│   │       ├── cached_decoder_layer.py     # 带缓存的解码器层
│   │       └── cached_multihead_attention.py # 带缓存的多头注意力
│   └── core/
│       ├── constants.py            # 全局常量
│       ├── utils.py                # 统计工具
│       └── tdr_guard.py            # CUDA 同步守卫
├── scripts/                        # 数据处理脚本
├── data/                           # 语料、词表、样本
├── models/                         # 模型权重与检查点
└── logs/                           # 训练日志与 TensorBoard
```

## 环境要求

- Python 3.13
- CUDA 兼容 GPU（显存 >= 24GB）
- 依赖：PyTorch 2.10.0、SentencePiece、TensorBoard、sacrebleu、NumPy、tqdm

安装依赖：

```bash
uv sync
```

## 使用

训练：

```bash
uv run python -m word_ferry.main
```

交互式推理：

```bash
uv run python -m word_ferry.tester
```

## 相关文章

- [衔言渡意：中英法，六个方向](https://fallingangel.net/archives/xian-yan-du-yi-zhong-ying-fa-liu-ge-fang-xiang)
- [通用 Tokenizer 评估方案——从项目专用到任务无关](https://fallingangel.net/archives/tong-yong-tokenizer-ping-gu-fang-an----cong-xiang-mu-zhuan-yong-dao-ren-wu-wu-guan)
- [从数据反推模型架构：一个小模型训练的经验公式](https://fallingangel.net/archives/cong-shu-ju-fan-tui-mo-xing-jia-gou-yi-ge-xiao-mo-xing-xun-lian-de-jing-yan-gong-shi)
- [模型容量不够——下一步是加层还是加宽？](https://fallingangel.net/archives/mo-xing-rong-liang-bu-gou----xia-yi-bu-shi-jia-ceng-huan-shi-jia-kuan)
- [训练控制的统计化——当 loss 和评估分数开始撒谎](https://fallingangel.net/archives/xun-lian-kong-zhi-de-tong-ji-hua----dang-loss-he-ping-gu-fen-shu-kai-shi-sa-huang)
- [KV Cache 实现手记——高估了，低估了，然后搞懂了](https://fallingangel.net/archives/kv-cache-shi-xian-shou-ji----gao-gu-liao-di-gu-liao-ran-hou-gao-dong-liao)
- [从 Loss 到微分，模型是怎么学习的？](https://fallingangel.net/archives/cong-loss-dao-wei-fen-mo-xing-shi-zen-me-xue-xi-de)
- [NLP 评估指标：你的模型到底好在哪](https://fallingangel.net/archives/nlpping-gu-zhi-biao-ni-de-mo-xing-dao-di-hao-zai-na)