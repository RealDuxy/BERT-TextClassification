# BERT-TextClassification



## 项目介绍

该部分项目采用数据集为[中文文本分类数据集THUCNews](http://thuctc.thunlp.org/)。

THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。我们在原始新浪新闻分类体系的基础上，重新整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。



## 环境依赖

Pytorch 1.4.0

Transformers 3.4.0



## 项目模型

#### 2020.11.24

BERT-wwm-base全冻结 + TextCNN

Epoch=4, lr=1e-3的结果，取用最后一层所有的token。

![image-20201125115552678](https://tva1.sinaimg.cn/large/0081Kckwly1gl1a4y9lwdj30ut0u0q8y.jpg)

#### 2020.11.25

BERT-wwm-base + TextCNN

Epoch=4

bert_lr = 2e-5, cls_lr=1e-3的结果，取用最后一层所有的token。

![2111606266974_.pic_hd](https://tva1.sinaimg.cn/large/0081Kckwly1gl1a6srnwvj30ts0lcq69.jpg)

![2121606266974_.pic_hd](https://tva1.sinaimg.cn/large/0081Kckwly1gl1a764b14j30z50u0wj6.jpg)

