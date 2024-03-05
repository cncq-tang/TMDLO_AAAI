import os
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import TMDLO
from data import Multi_view_data
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training [default: 100]')  # 输入批次大小（即每次训练时输入的样本数量）
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train [default: 500]')  # 训练的周期数
    parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')  # 接受一个整数值作为将lambda值从0逐渐增加到1所需的周期数
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate')  # 学习率
    args = parser.parse_args()  # 解析命令行参数，并将解析结果存储在args变量中
    args.data_name = 'handwritten_6views'
    args.data_path = 'datasets/' + args.data_name
    args.dims = [[240], [76], [216], [47], [64], [6]]  # 6种视图的各自维度
    args.views = len(args.dims)  # 视图数

    # 训练集数据加载器
    train_loader = torch.utils.data.DataLoader(
        Multi_view_data(args.data_path, train=True), batch_size=args.batch_size,
        shuffle=True)  # Multi_view_data()自定义类,shuffle=True 表示在每个训练周期开始前对数据进行随机打乱
    # 测试集数据加载器
    test_loader = torch.utils.data.DataLoader(
        Multi_view_data(args.data_path, train=False), batch_size=args.batch_size, shuffle=False)
    N_mini_batches = len(train_loader)  # 加载器的每个周期的批次数量

    print('The number of epochs = %d' % args.epochs)
    print('The number of batches per epoch = %d' % N_mini_batches)
    print('The number of pictures per batch in each epoch = %d' % args.batch_size)
    print('The learning rate =%.4f' % args.lr)
    print('-----------------------------------------------------------------------')

    model = TMDLO(10, args.views, args.dims)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)  # weight_decay为正则化L2强度

    model.cuda()


    def train(epoch):
        model.train()
        loss_meter = AverageMeter()
        for batch_idx, (data, target) in enumerate(train_loader):  # 每个批次的数据
            for v_num in range(len(data)):  # 当前批次的数据
                data[v_num] = Variable(data[v_num].cuda())
            target = Variable(target.long().cuda())
            # refresh the optimizer
            optimizer.zero_grad()  # 清零了优化器中所有参数的梯度。这是因为在 PyTorch 中，梯度默认会在反向传播过程中累积，所以在每次迭代之前需要将其清零
            evidences, evidence_a, loss = model(data, target, epoch)  # 调用模型进行前向传播，并计算模型输出、损失和其它返回值
            # compute gradients and take step
            loss.backward()  # 执行反向传播，计算损失函数关于模型参数的梯度
            optimizer.step()  # 根据损失函数的梯度更新模型参数
            loss_meter.update(loss.item())  # 一行将当前批次的损失值添加到 loss_meter 中，并更新统计信息。这样可以跟踪整个训练过程中损失值的变化情况。
            print('The loss value of batch %d of the %d training epoch = %.4f' % (batch_idx, epoch, loss.item()))
        print('The average loss value for the %d training epoch =%.4f ' % (epoch, loss_meter.avg))
        print('----------------------------------------------------------')


    def test(epoch):
        model.eval()  # 将模型设置为评估模式
        loss_meter = AverageMeter()
        correct_num, data_num = 0, 0
        for batch_idx, (data, target) in enumerate(test_loader):
            for v_num in range(len(data)):
                data[v_num] = Variable(data[v_num].cuda())
            data_num += target.size(0)
            with torch.no_grad():
                target = Variable(target.long().cuda())
                evidences, evidence_a, loss = model(data, target, epoch)
                _, predicted = torch.max(evidence_a.data, 1)  # 每行中的最大值以及对应的索引
                correct_num += (predicted == target).sum().item()
                loss_meter.update(loss.item())

        return loss_meter.avg, correct_num / data_num


    for epoch in range(1, args.epochs + 1):
        train(epoch)

    test_loss, acc = test(epoch)
    print('The test epoch is %d' % epoch)
    print('The average loss of the test set is ====> loss: {:.4f}'.format(test_loss))
    print('The classification accuracy of the test set is ====> acc: {:.4f}'.format(acc))
