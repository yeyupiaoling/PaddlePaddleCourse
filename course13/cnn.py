import paddle


class CNN(paddle.nn.Layer):
    def __init__(self):
        super(CNN, self).__init__()
        # 定义每个网络的结构
        self.conv1 = paddle.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.conv2 = paddle.nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5)
        self.pool1 = paddle.nn.MaxPool2d(kernel_size=2, stride=2)
        self.input_dim = 50 * 4 * 4
        self.fc = paddle.nn.Linear(in_features=self.input_dim, out_features=10)

def forward(self, inputs):
    # 把每个网络组合在一起
    x = self.conv1(inputs)
    x = paddle.nn.functional.relu(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = paddle.nn.functional.relu(x)
    x = self.pool1(x)
    x = paddle.flatten(x, start_axis=1,stop_axis=-1)
    x = self.fc(x)
    x = paddle.nn.functional.softmax(x)
    return x