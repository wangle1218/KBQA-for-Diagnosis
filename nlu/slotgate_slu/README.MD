## keras 复现SlotGated-SLU 模型

[Slot-Gated Modeling for Joint Slot Filling and Intent Prediction](https://www.aclweb.org/anthology/N18-2118/)，[官方开源代码](https://github.com/MiuLab/SlotGated-SLU)

这是2018年的一篇论文，我在[这篇文章](https://mp.weixin.qq.com/s/lLSuMW2VvBKQyf_POi-ENA)里作了详细的介绍，这是一个意图识别和槽位填充的联合训练模型，由于还算比较有创新，于是我尝试使用这个模型来跑实验。官方开源代码使用 tf1.4 版本写的，版本较老，另外源代码的代码结构写的比较随意，不适合在实际工作中使用，其次可读性也不强；因此我花了点时间用 keras 来复现了这个模型。

下面两个图分别是我复现的模型在 atis 和 snips 两个数据集上的评估结果，基本上和原论文的实验结果差不多。另外由于我没有使用预训练的词向量，因此embedding 是随机初始化的，在snips数据集上让embedding 的参数可训练，相比于不训练在槽位填充任务上能提升三四个点，而且收敛的更快；我想如果用预训练的词向量效果应该会更好。

<img src="./img/atis-res.png" alt="avatar" style="zoom:60%;" />



<img src="./img/snips-res.png" alt="avatar" style="zoom:60%;" />

从[这篇文章](https://mp.weixin.qq.com/s/lLSuMW2VvBKQyf_POi-ENA) 里可以看到本模型的结构其实还是比较简单的，主要是需要实现 intent attention、slot attention 和 slot gated 三个模块，其中 intent attention 在论文里没有详细的介绍，对照官方的源代码我认为这应该就是我们最常用的那种 Attention 了，它的结构应该是符合下图所示的：

<img src="./img/attn1.png" alt="avatar" style="zoom:90%;" />

句子经过双向LSTM 的编码，输出 shape 为 [batch_size, maxlen, 2*lstm_units]，经过 intent attention 后的输出的shape 为 [batch_size,maxlen]，这个变换其实和 globa mean pooling 差不多，只不过不是做平均的加权，而是通过计算一个注意力权重系数来加权的。

slot attention 和上述注意力不同，经过该注意力机制后的输出 shape 仍然为 [batch_size, maxlen, 2*lstm_units]，没有降维，这个在论文里有公式可供参考，具体我就不多说了，也可以去看我的复现代码。

本模型创新点 slot gated 在论文里也有详细的公式介绍，也可去看我的复现代码。

### 如何运行本项目代码

1、下载本仓库代码，进入项目所在文件夹

2、使用脚本 `train.py` 进行训练和评估

```python
# 使用 full_attention 参数确定是否使用 slot attention
# 使用 dataset 参数确定使用哪一个数据集做实验，可选的有 snips 和 atis
# 使用 max_features 参数指定词表大小，snips 数据集建议11250，atis 数据集建议750
python train.py --max_epochs=80 --full_attention=True --dataset=snips --max_features=11250
```

