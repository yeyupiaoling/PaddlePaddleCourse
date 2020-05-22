import paddle


class CNN(paddle.nn.Layer):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义每个网络的结构
        self.conv1 = paddle.nn.Conv2D(num_channels=1, num_filters=20, filter_size=5, act="relu")
        self.conv2 = paddle.nn.Conv2D(num_channels=20, num_filters=50, filter_size=5, act="relu")
        self.pool1 = paddle.nn.Pool2D(pool_size=2, pool_type='max', pool_stride=2)
        self.input_dim = 50 * 4 * 4
        self.fc = paddle.nn.Linear(input_dim=self.input_dim, output_dim=10, act='softmax')

    def forward(self, inputs):
        # 把每个网络组合在一起
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = paddle.reshape(x, shape=[-1, self.input_dim])
        x = self.fc(x)
        return x


