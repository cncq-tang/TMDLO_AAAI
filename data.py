import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class Multi_view_data(Dataset):
    """
    load multi-view data
    """

    def __init__(self, root, train=True):
        """
        :param root:data name and path
        :param train:load training set or test set
        """
        super(Multi_view_data, self).__init__()
        self.root = root
        self.train = train

        # 构造数据文件的路径，通过在 root路径名后面加上.mat 扩展名。
        data_path = self.root + '.mat'

        # 将 .mat格式 数据读入
        dataset = sio.loadmat(data_path)

        # 计算视角数目，通过数据集的长度来推断。
        view_number = int((len(dataset) - 5) / 2)

        # 初始化一个空字典，用于存储不同视角的数据。
        self.X = dict()

        # 判断是训练还是测试
        if train:
            for v_num in range(view_number):
                self.X[v_num] = normalize(dataset['x' + str(v_num + 1) + '_train'])  # x1_train x2_train x3_train
            y = dataset['gt_train']  # 训练集对应类别
        else:
            for v_num in range(view_number):
                self.X[v_num] = normalize(dataset['x' + str(v_num + 1) + '_test'])
            y = dataset['gt_test']  # 测试集对应类别

        # 如果最小值是1，意味着标签的范围可能是从1开始的。为了使标签从0开始，下面的代码会将所有标签数字减去1。
        if np.min(y) == 1:
            y = y - 1

        # 创建一个与 y 具有相同长度的全零数组。这里使用了 y.shape[0] 来获取 y 的长度。
        tmp = np.zeros(y.shape[0])  # y的行数
        y = np.reshape(y, np.shape(tmp))  # 形状为1*400的数组
        self.y = y

    """
    Gets the data and categories for the corresponding index
    """
    def __getitem__(self, index):
        data = dict()
        for v_num in range(len(self.X)):
            data[v_num] = (self.X[v_num][index]).astype(np.float32)  # date包括多个视角的数据，x1[index],x2[index],...
        target = self.y[index]  # 对应类别
        return data, target

    def __len__(self):
        return len(self.X[0])


def normalize(x, min=0):
    """
    :param x:待归一化数据
    :param min:归一化数据范围选择，默认为[0,1]范围
    return:归一化后的Numpy数组
    """
    if min == 0:  # 如果 min 为 0，则使用 [0, 1] 的范围进行归一化
        scaler = MinMaxScaler((0, 1))
    else:  # 如果 min 为 -1，则使用 [-1, 1] 的范围进行归一化
        scaler = MinMaxScaler((-1, 1))

    # 使用 MinMaxScaler 对数据进行归一化
    norm_x = scaler.fit_transform(x)

    # 返回的值类型是 NumPy 数组
    return norm_x
