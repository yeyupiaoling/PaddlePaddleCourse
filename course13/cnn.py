import paddle.fluid as fluid


class CNN(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(CNN, self).__init__(name_scope)
        # 定义每个网络的结构
        self.conv1 = fluid.dygraph.Conv2D(self.full_name(), 20, 5, act="relu")
        self.conv2 = fluid.dygraph.Conv2D(self.full_name(), 50, 5, act="relu")
        self.pool1 = fluid.dygraph.Pool2D(self.full_name(), pool_size=2, pool_type='max',
                                          pool_stride=2)
        self.fc = fluid.dygraph.FC(self.full_name(), size=10, act='softmax')

    def forward(self, inputs):
        # 把每个网络组合在一起
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.fc(x)
        return x