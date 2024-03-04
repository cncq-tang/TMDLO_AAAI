import torch
import torch.nn as nn


# 神经网络结构
class Classifier(nn.Module):
    """
    classifier_dims:神经网络各层的维度，即数据集各个视角维度:[[240], [76], [216], [47], [64], [6]]
    classes:输出的类别数量
    """

    def __init__(self, classifier_dims, classes):
        super(Classifier, self).__init__()
        self.num_layers = len(classifier_dims)
        self.fc = nn.ModuleList()  # 创建了一个空的 nn.ModuleList() 对象 self.fc，用于存储神经网络的所有线性层
        for i in range(self.num_layers - 1):
            # 添加线性层，输入大小为classifier_dims[i]，输出大小为classifier_dims[i + 1]，默认使用偏置量
            self.fc.append(nn.Linear(classifier_dims[i], classifier_dims[i + 1]))
        self.fc.append(nn.Linear(classifier_dims[self.num_layers - 1], classes))  # 最后一层分类输出层
        self.fc.append(nn.Softplus())  # 激活函数层，Relu的替代方案

    """
    x:神经网络的输入数据
    return:神经网络的最终输出，即经过所有线性层和激活函数后的结果
    """

    def forward(self, x):
        h = self.fc[0](x)  # 把输入X传递给第一个线性层
        for i in range(1, len(self.fc)):
            h = self.fc[i](h)
        return h
