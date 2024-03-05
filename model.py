import torch
import torch.nn as nn


# 模型结构
class TMDLO(nn.Module):
    def __init__(self, classes, views, classifier_dims):
        """
        :param classes:分类的类别数量
        :param views:视图数量
        :param classifier_dims:神经网络各层维度
        """
        super(TMDLO, self).__init__()
        self.views = views
        self.classes = classes
        self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in
                                          range(self.views)])  # 多个视图的分类器。每个视图都有一个独立的分类器，用于处理对应视图的特征

        def Opinion_Aggregation(self, evidences):
            """
            :param evidence:一个样本的所有视图的e参数字典
            :return:可信的累计分类意见结果
            """
            # 所有视图的各个类的分类证据和
            e_kM = [0.0] * classes
            # 对 k 类的先验偏好，如果没有偏好则默认为 1/k
            a_kM = [1.0 / classes] * classes

            # 计算每个类别的分类证据和
            for evidence in evidences.values():
                for k, e in enumerate(evidence):
                    e_kM[k] += e

            # 计算所有视图的每个类别的狄利克雷参数累计和
            alpha_kM = [e_k + 1 for e_k in e_kM]
            # 计算 Dirichlet strength
            S_M = sum(e_kM) + classes

            # 计算所有视图的每个类别的分类概率
            b_kM = [e_k / S_M for e_k in e_kM]
            # 计算累计意见的不确定性
            U_M = 1.0 - sum(b_kM)

            return alpha_kM, b_kM, U_M, a_kM


# 神经网络结构
class Classifier(nn.Module):
    def __init__(self, classifier_dims, classes):
        """
        :param classifier_dims: 神经网络各层的维度，即数据集各个视角维度:[[240], [76], [216], [47], [64], [6]]
        :param classes: 输出的类别数量
        """
        super(Classifier, self).__init__()
        self.num_layers = len(classifier_dims)
        self.fc = nn.ModuleList()  # 创建了一个空的 nn.ModuleList() 对象 self.fc，用于存储神经网络的所有线性层
        for i in range(self.num_layers - 1):
            # 添加线性层，输入大小为classifier_dims[i]，输出大小为classifier_dims[i + 1]，默认使用偏置量
            self.fc.append(nn.Linear(classifier_dims[i], classifier_dims[i + 1]))
        self.fc.append(nn.Linear(classifier_dims[self.num_layers - 1], classes))  # 最后一层分类输出层
        self.fc.append(nn.Softplus())  # 激活函数层，Relu的替代方案

    def forward(self, x):
        """
        :param x: 神经网络的输入数据
        :return: 神经网络的最终输出，即经过所有线性层和激活函数后的结果
        """
        h = self.fc[0](x)  # 把输入X传递给第一个线性层
        for i in range(1, len(self.fc)):
            h = self.fc[i](h)
        return h
