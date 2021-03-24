# Dual-path
针对论文Dual-Path Convolutional Image-Text Embeddings with Instance Loss的复现

作者开源代码(matlab)：https://github.com/layumi/Image-Text-Embedding

代码是在https://github.com/hqq624308/Dual-path 的基础上改进，修复了一些bugs

#### Get Start
1. 下载flickr30k数据集，链接：https://pan.baidu.com/s/1E1mfsuoDsc56OZtmuNnJrQ  提取码：1ati 
2. git clone https://github.com/ChronousZhang/Dual-path.git
3. 下载word2vector模型，用于将文本转换为词向量，链接：https://pan.baidu.com/s/1Y9D73eSFlLxFh4SmrTfOew 提取码：n4rs 
4. mkdir /data/yourUser/flickr30k/Dual-path 并将下载好的flickr30k数据集放到/data/yourUser/flickr30k中
5. 解压flickr30k数据集：tar -xzvf flickr30k.tar.gz   tar -xvf flickr30k-images.tar
6. 运行解压后的flickr30k_test.py文件可以查看图像和对应的自然语言描述
7. 运行python text2vec.py将数据集切分为train,val和test并转换为词向量，其中test取最后1000张图像和其对应的描述，val取倒数第二个1000张图像和其对应的描述，其余为train，与论文保持一致
8. 如果希望跳过7，也可直接下载处理处理后的结果，链接：https://pan.baidu.com/s/1_3s5FTgm51xr3uskgguZhg 提取码：ah57 
9. 运行python train_stageI_instance.py使用instance loss(每张图像当作一个类比下的CE损失)进行第一阶段模型训练(感觉收敛不是很快，可能类别太多吧，第一次训练这么大的分类任务)
10. 运行train_stageII_rank_instance.py使用instance loss和rank loss联合训练模型第二阶段
11. 自动运行步骤7，9，10只需要bash run.sh即可

#### Attention
1. 测试阶段仅测试了“以文搜图”，因此text_acc始终为零，log中CMC和mAP也都是“以文搜图”的精度
