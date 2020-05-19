import paddle.fluid as fluid
import paddle
import paddle.dataset.uci_housing as uci_housing
import numpy

# 定义一个简单的线性网络
x = fluid.data(name='x', shape=[None, 13], dtype='float32')
hidden = fluid.layers.fc(input=x, size=100, act='relu')
hidden = fluid.layers.fc(input=hidden, size=100, act='relu')
net = fluid.layers.fc(input=hidden, size=1, act=None)

# 获取预测程序
infer_program = fluid.default_main_program().clone(for_test=True)

# 定义损失函数
y = fluid.data(name='y', shape=[None, 1], dtype='float32')
cost = fluid.layers.square_error_cost(input=net, label=y)
avg_cost = fluid.layers.mean(cost)

# 定义优化方法
optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01)
opts = optimizer.minimize(avg_cost)

# 创建一个使用CPU的执行器
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 从paddle接口中获取房价数据集，使用房价数据进行训练和测试
train_reader = paddle.batch(reader=uci_housing.train(), batch_size=128)
test_reader = paddle.batch(reader=uci_housing.test(), batch_size=128)

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])

# 开始训练和测试
for pass_id in range(300):
    # 开始训练并输出最后一个batch的损失值
    train_cost = 0
    for batch_id, data in enumerate(train_reader()):
        train_cost = exe.run(program=fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[avg_cost])
    print("Pass:%d, Cost:%0.5f" % (pass_id, train_cost[0][0]))

# 开始测试
for data in test_reader():
    infer_data = numpy.array([data[0] for data in data]).astype("float32")
    infer_label = numpy.array([data[1] for data in data]).astype("float32")
    infer_result = exe.run(program=infer_program,
                           feed={x.name: infer_data},
                           fetch_list=[net])
    # 把每个预测结果都输出
    for i in range(len(infer_label)):
        print('预测结果：%f, 真实结果：%f' % (infer_result[0][i][0], infer_label[i][0]))
