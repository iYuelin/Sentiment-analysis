# Sentiment-analysis
Sentiment analysis on movie reviews：
1. First, preprocess the training data and validation data set, delete punctuation marks, stop words and irregular words. Reuse the processed training data and verification data to build a vocabulary dictionary. And convert each comment text into a list of numbers. Since the network input data must be the same latitude, according to the statistics of the processed text length, 1000 is taken as the uniform text length.
2. Flatten compresses the data of the input layer into one-dimensional data, the transition from the convolutional layer to the fully connected layer.
3. Dense is a fully connected layer, which acts as a "classifier" in the entire convolutional neural network.
4. The Dropout layer is used to prevent over-fitting. Each time the parameters are updated during the training process, the input neurons are randomly disconnected at a certain probability (rate).
5. The model finally determines the weight of each layer after 20 iterations.

基于电影评价的文本情感分析：
1、对训练数据和验证数据集进行预处理，例如：删除标点符号、停用词和不规则词等，利用处理后的训练文本和验证文本数据构建词汇词典并将每个评论文本转换为数字列表。 由于网络输入数据必须保持同一维度，因此，根据处理后的文本长度统计，取1000作为统一的文本长度；
2.Flatten将输入层的数据压缩成一维数据，从卷积层过渡到全连接层。
3.Dense是全连接层，在整个卷积神经网络中充当“分类器”。
4. Dropout层用于防止过拟合。 在训练过程中每次更新参数，输入神经元都会以一定的概率随机断开。
5.设置模型迭代次数为20，从而确定每一层的权重。
